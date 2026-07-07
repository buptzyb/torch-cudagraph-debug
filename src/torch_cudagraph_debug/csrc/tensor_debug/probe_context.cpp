#include "tensor_debug/probe_context.h"

#include "common/cuda_utils.h"
#include "tensor_debug/check.h"
#include "tensor_debug/replay_counter.h"
#include "tensor_debug/tensor_format.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace torch_cudagraph_debug::tensor_debug {
namespace {

std::mutex registry_mutex;
std::unordered_map<uint64_t, std::shared_ptr<ProbeContext>> registry;
std::atomic<uint64_t> next_context_id{1};

void unregister_context(uint64_t id) {
    std::lock_guard<std::mutex> guard(registry_mutex);
    registry.erase(id);
}

int64_t tensor_nbytes(const torch::Tensor& tensor) {
    return tensor.numel() * tensor.element_size();
}

bool same_shape(const std::vector<int64_t>& expected_shape, at::IntArrayRef actual_shape) {
    if (expected_shape.size() != static_cast<size_t>(actual_shape.size())) {
        return false;
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] != actual_shape[static_cast<int64_t>(i)]) {
            return false;
        }
    }
    return true;
}

const ExpectedTensorConfig* find_expected(
    const CheckActionConfig& check,
    uint64_t order,
    const std::string& observation_name,
    uint64_t invocation_index) {
    if (!check.keyed) {
        return order < check.expected.size()
            ? &check.expected[static_cast<size_t>(order)]
            : nullptr;
    }
    for (const ExpectedTensorConfig& expected : check.expected) {
        if (expected.observation_name == observation_name &&
            expected.invocation_index == invocation_index) {
            return &expected;
        }
    }
    return nullptr;
}

torch::Tensor observation_to_tensor(const TensorObservationData& observation) {
    auto options = torch::TensorOptions().device(torch::kCPU).dtype(observation.dtype);
    torch::Tensor tensor = torch::empty(observation.shape, options);
    if (observation.nbytes > 0) {
        std::memcpy(tensor.data_ptr(), observation.bytes.data(), observation.nbytes);
    }
    return tensor;
}

}  // namespace

ProbeContext::ProbeContext(
    uint64_t id,
    std::string name,
    std::vector<ActionConfig> actions,
    torch::Tensor replay_index,
    NonContiguousPolicy non_contiguous,
    ProbeMode mode)
    : id_(id),
      name_(std::move(name)),
      actions_(std::move(actions)),
      replay_index_(std::move(replay_index)),
      non_contiguous_(non_contiguous),
      mode_(mode) {
    if (!replay_index_.defined() || !replay_index_.is_cuda() ||
        replay_index_.scalar_type() != at::kLong || replay_index_.numel() != 1 ||
        !replay_index_.is_contiguous()) {
        throw std::runtime_error(
            "tensor debug replay_index must be a contiguous CUDA int64 scalar");
    }
    replay_index_device_ = replay_index_.get_device();

    for (const ActionConfig& action : actions_) {
        if (action.kind == ActionConfig::Kind::Record && action.record.enabled) {
            has_record_action_ = true;
        } else if (
            (action.kind == ActionConfig::Kind::Print && action.print.enabled) ||
            (action.kind == ActionConfig::Kind::Check && action.check.enabled)) {
            has_callback_actions_ = true;
        }
    }

    c10::cuda::CUDAGuard device_guard(replay_index_.device());
    try {
        CaptureModeGuard capture_mode_guard(cudaStreamCaptureModeRelaxed);
        if (has_callback_actions_) {
            TCGD_CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&replay_index_staging_),
                sizeof(*replay_index_staging_)));
            *replay_index_staging_ = 0;
            TCGD_CUDA_CHECK(cudaEventCreateWithFlags(
                &replay_index_ready_event_, cudaEventDisableTiming));
        }
    } catch (...) {
        release_resources_noexcept();
        throw;
    }
}

ProbeContext::~ProbeContext() {
    release_resources_noexcept();
}

torch::Tensor ProbeContext::enqueue(
    const torch::Tensor& tensor,
    const std::string& observation_name,
    uint64_t invocation_index) {
    ensure_open();
    if (observation_name.empty()) {
        throw std::runtime_error("observation name must be non-empty");
    }

    if ((!tensor.defined() || !tensor.is_cuda()) && mode_ == ProbeMode::Capture) {
        return tensor;
    }
    if (!tensor.defined() || !tensor.is_cuda()) {
        validate_tensor(tensor);
    }

    c10::cuda::CUDAGuard device_guard(tensor.device());
    const c10::cuda::CUDAStream current_stream =
        c10::cuda::getCurrentCUDAStream(tensor.get_device());
    cudaStream_t stream = current_stream.stream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    TCGD_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;
    if (mode_ == ProbeMode::Capture && !is_capturing) {
        return tensor;
    }

    if (!is_capturing) {
        validate_eager_stream(stream);
    }
    const uint64_t capture_id = is_capturing ? capture_id_for_stream(stream) : 0;
    const uint64_t order = next_slot_index(is_capturing, capture_id);

    validate_tensor(tensor);
    validate_check_actions(
        tensor,
        order,
        observation_name,
        invocation_index);

    torch::Tensor source = source_tensor_for_enqueue(tensor);
    const size_t nbytes = static_cast<size_t>(tensor_nbytes(tensor));

    TensorSlot& slot =
        ensure_slot(
            tensor,
            nbytes,
            order,
            observation_name,
            invocation_index,
            is_capturing);
    if (!tensor.is_contiguous() && is_capturing) {
        slot.source_owners.push_back(source);
    } else if (!tensor.is_contiguous()) {
        c10::cuda::CUDACachingAllocator::recordStream(
            source.storage().data_ptr(),
            current_stream);
    }

    if (is_capturing && order == 0) {
        launch_increment_replay_counter(
            replay_index_.data_ptr<int64_t>(), stream);
        if (has_callback_actions_) {
            TCGD_CUDA_CHECK(cudaMemcpyAsync(
                replay_index_staging_,
                replay_index_.data_ptr<int64_t>(),
                sizeof(*replay_index_staging_),
                cudaMemcpyDeviceToHost,
                stream));
            TCGD_CUDA_CHECK(cudaEventRecord(replay_index_ready_event_, stream));
        }
    }

    if (nbytes > 0) {
        TCGD_CUDA_CHECK(cudaMemcpyAsync(
            slot.staging,
            source.data_ptr(),
            nbytes,
            cudaMemcpyDeviceToHost,
            stream));
    }
    if (has_callback_actions_) {
        if (is_capturing && order > 0) {
            TCGD_CUDA_CHECK(cudaStreamWaitEvent(
                stream, replay_index_ready_event_, 0));
        }
        std::unique_ptr<CallbackPayload> payload = make_payload(
            tensor,
            slot.staging,
            nbytes,
            order,
            observation_name,
            invocation_index,
            is_capturing);
        if (is_capturing) {
            CallbackPayload* raw = payload.get();
            TCGD_CUDA_CHECK(
                cudaLaunchHostFunc(stream, &ProbeContext::host_callback, raw));
            captured_payloads_.push_back(std::move(payload));
        } else {
            CallbackPayload* raw = payload.release();
            eager_callbacks_in_flight_.fetch_add(1, std::memory_order_release);
            const cudaError_t status =
                cudaLaunchHostFunc(stream, &ProbeContext::host_callback, raw);
            if (status != cudaSuccess) {
                eager_callbacks_in_flight_.fetch_sub(1, std::memory_order_release);
                payload.reset(raw);
                TCGD_CUDA_CHECK(status);
            }
        }
    }
    return tensor;
}

pybind11::list ProbeContext::observations(std::optional<uint64_t> replay_index) {
    uint64_t resolved_replay_index = 0;
    if (replay_index.has_value()) {
        resolved_replay_index = *replay_index;
    } else if (replay_index_staging_ != nullptr) {
        const int64_t staged_replay_index = *replay_index_staging_;
        resolved_replay_index = staged_replay_index > 0
            ? static_cast<uint64_t>(staged_replay_index)
            : 0;
    } else {
        throw std::runtime_error(
            "observations requires replay_index when callback counter staging is unavailable");
    }

    std::vector<TensorObservationData> ordered;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ordered.reserve(slots_.size());
        for (const TensorSlot& slot : slots_) {
            if (!slot.observation.valid) {
                continue;
            }
            TensorObservationData observation = slot.observation;
            observation.replay_index = observation.captured ? resolved_replay_index : 0;
            observation.bytes.clear();
            observation.bytes.resize(observation.nbytes);
            if (observation.nbytes > 0 && slot.staging != nullptr) {
                std::memcpy(observation.bytes.data(), slot.staging, observation.nbytes);
            }
            ordered.push_back(std::move(observation));
        }
    }

    pybind11::list result;
    for (const TensorObservationData& observation : ordered) {
        pybind11::dict item;
        item["name"] = observation.name;
        item["order"] = observation.order;
        item["replay_index"] = observation.replay_index;
        item["invocation_index"] = observation.invocation_index;
        item["shape"] = observation.shape;
        item["device"] = observation.device;
        item["tensor"] = observation_to_tensor(observation);
        result.append(item);
    }
    return result;
}

pybind11::dict ProbeContext::check_status() {
    std::lock_guard<std::mutex> guard(mutex_);
    pybind11::dict result;
    result["ok"] = !check_failed_;
    result["message"] = failure_message_;
    result["replay_index"] = failure_replay_index_;
    result["order"] = failure_order_;
    if (failure_name_.empty()) {
        result["name"] = pybind11::none();
    } else {
        result["name"] = failure_name_;
    }
    result["invocation_index"] = failure_invocation_index_;
    return result;
}

pybind11::dict ProbeContext::debug_resource_counts() const {
    std::lock_guard<std::mutex> guard(mutex_);
    size_t source_owner_count = 0;
    for (const TensorSlot& slot : slots_) {
        source_owner_count += slot.source_owners.size();
    }

    pybind11::dict result;
    result["captured_payloads"] = captured_payloads_.size();
    result["eager_callbacks_in_flight"] =
        eager_callbacks_in_flight_.load(std::memory_order_acquire);
    result["source_owners"] = source_owner_count;
    result["retired_staging"] = retired_staging_.size();
    return result;
}

void ProbeContext::reclaim_retired_staging() {
    std::vector<void*> retired;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        retired.swap(retired_staging_);
    }
    for (void* ptr : retired) {
        if (ptr != nullptr) {
            TCGD_CUDA_CHECK(cudaFreeHost(ptr));
        }
    }
}

void ProbeContext::close() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closed_) {
            return;
        }
        const uint64_t in_flight =
            eager_callbacks_in_flight_.load(std::memory_order_acquire);
        if (in_flight != 0) {
            std::ostringstream oss;
            oss << "cannot close tensor debug probe " << name_
                << " while " << in_flight
                << " eager callback(s) are pending; synchronize first";
            throw std::runtime_error(oss.str());
        }
        closed_ = true;
    }

    release_resources_noexcept();
    captured_payloads_.clear();
    slots_.clear();
    replay_index_ = torch::Tensor();
    unregister_context(id_);
}

void ProbeContext::on_callback(const CallbackPayload& payload) noexcept {
    try {
        uint64_t replay_index = 0;
        uint64_t schedule_index = 0;
        const uint64_t order = payload.order;
        const std::string& observation_name = payload.name;
        const uint64_t invocation_index = payload.invocation_index;
        if (payload.captured && replay_index_staging_ != nullptr) {
            const int64_t captured_replay_index = *replay_index_staging_;
            replay_index = captured_replay_index > 0
                ? static_cast<uint64_t>(captured_replay_index)
                : 0;
            schedule_index = replay_index;
        } else {
            std::lock_guard<std::mutex> guard(mutex_);
            schedule_index = ++eager_callback_count_;
        }

        for (const ActionConfig& action : actions_) {
            if (action.kind == ActionConfig::Kind::Print && action.print.enabled) {
                if (schedule_index % static_cast<uint64_t>(action.print.every) == 0) {
                    const std::string formatted = format_tensor_bytes(
                        payload.staging,
                        payload.numel,
                        payload.dtype,
                        payload.shape,
                        action.print.max_items,
                        action.print.summary);
                    std::fprintf(
                        stderr,
                        "[torch-cudagraph-debug:%s] replay=%llu observation=%s[%llu] order=%llu dtype=%s %s\n",
                        name_.c_str(),
                        static_cast<unsigned long long>(replay_index),
                        observation_name.c_str(),
                        static_cast<unsigned long long>(invocation_index),
                        static_cast<unsigned long long>(order),
                        scalar_type_name(payload.dtype).c_str(),
                        formatted.c_str());
                    std::fflush(stderr);
                }
            } else if (action.kind == ActionConfig::Kind::Check && action.check.enabled) {
                const ExpectedTensorConfig* expected_ptr = find_expected(
                    action.check, order, observation_name, invocation_index);
                if (expected_ptr == nullptr) {
                    std::ostringstream oss;
                    oss << "CheckAction expected values for probe " << name_
                        << " have no tensor for observation " << observation_name
                        << "[" << invocation_index << "] at order " << order;
                    set_failure(
                        replay_index,
                        static_cast<int64_t>(order),
                        observation_name,
                        static_cast<int64_t>(invocation_index),
                        oss.str());
                    continue;
                }
                const ExpectedTensorConfig& expected = *expected_ptr;
                if (payload.dtype != expected.expected_dtype ||
                    payload.shape != expected.expected_shape ||
                    payload.numel != expected.expected_numel) {
                    std::ostringstream oss;
                    oss << "check metadata mismatch for probe " << name_
                        << " observation " << observation_name << "["
                        << invocation_index << "] at order " << order;
                    set_failure(
                        replay_index,
                        static_cast<int64_t>(order),
                        observation_name,
                        static_cast<int64_t>(invocation_index),
                        oss.str());
                    continue;
                }
                CheckResult result = check_tensor_bytes(
                    payload.staging,
                    expected.expected_bytes.data(),
                    payload.numel,
                    payload.dtype,
                    action.check.rtol,
                    action.check.atol,
                    action.check.equal_nan);
                if (!result.ok) {
                    std::ostringstream oss;
                    oss << "probe " << name_ << " observation "
                        << observation_name << "[" << invocation_index
                        << "] at order " << order << " " << result.message;
                    set_failure(
                        replay_index,
                        static_cast<int64_t>(order),
                        observation_name,
                        static_cast<int64_t>(invocation_index),
                        oss.str());
                }
            }
        }
    } catch (const std::exception& exc) {
        set_failure(
            0,
            static_cast<int64_t>(payload.order),
            payload.name,
            static_cast<int64_t>(payload.invocation_index),
            std::string("host callback error for probe ") + name_ + ": " + exc.what());
    } catch (...) {
        set_failure(
            0,
            static_cast<int64_t>(payload.order),
            payload.name,
            static_cast<int64_t>(payload.invocation_index),
            std::string("unknown host callback error for probe ") + name_);
    }
}

void ProbeContext::ensure_open() const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (closed_) {
        throw std::runtime_error("tensor debug probe is closed: " + name_);
    }
}

void ProbeContext::validate_tensor(const torch::Tensor& tensor) const {
    if (!tensor.defined()) {
        throw std::runtime_error("tensor debug probe input must be defined");
    }
    if (!tensor.is_cuda()) {
        throw std::runtime_error("tensor debug probe input must be a CUDA tensor");
    }
    if (tensor.get_device() != replay_index_device_) {
        std::ostringstream oss;
        oss << "tensor debug probe " << name_ << " was created on cuda:"
            << replay_index_device_ << " but received a tensor on cuda:"
            << static_cast<int>(tensor.get_device());
        throw std::runtime_error(oss.str());
    }
    if (!tensor.is_contiguous() && non_contiguous_ == NonContiguousPolicy::Error) {
        throw std::runtime_error(
            "tensor debug probe input must be contiguous; pass non_contiguous=\"copy\" "
            "to allow an internal debug-only contiguous CUDA copy");
    }
    scalar_type_size(tensor.scalar_type());
}

void ProbeContext::validate_check_actions(
    const torch::Tensor& tensor,
    uint64_t order,
    const std::string& observation_name,
    uint64_t invocation_index) const {
    for (const ActionConfig& action : actions_) {
        if (action.kind != ActionConfig::Kind::Check || !action.check.enabled) {
            continue;
        }
        const ExpectedTensorConfig* expected_ptr = find_expected(
            action.check, order, observation_name, invocation_index);
        if (expected_ptr == nullptr) {
            std::ostringstream oss;
            oss << "CheckAction expected values for probe " << name_
                << " have no tensor for observation " << observation_name
                << "[" << invocation_index << "] at order " << order;
            throw std::runtime_error(oss.str());
        }
        const ExpectedTensorConfig& expected = *expected_ptr;
        if (tensor.scalar_type() != expected.expected_dtype) {
            std::ostringstream oss;
            oss << "CheckAction expected dtype does not match probe input dtype"
                << " for observation " << observation_name << "["
                << invocation_index << "] at order " << order;
            throw std::runtime_error(oss.str());
        }
        if (!same_shape(expected.expected_shape, tensor.sizes())) {
            std::ostringstream oss;
            oss << "CheckAction expected shape does not match probe input shape"
                << " for observation " << observation_name << "["
                << invocation_index << "] at order " << order;
            throw std::runtime_error(oss.str());
        }
    }
}

uint64_t ProbeContext::capture_id_for_stream(cudaStream_t stream) const {
#if CUDART_VERSION >= 11030
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    TCGD_CUDA_CHECK(cudaStreamGetCaptureInfo(
        stream,
        &capture_status,
        &capture_id));
    if (capture_status == cudaStreamCaptureStatusNone) {
        return 0;
    }
    return static_cast<uint64_t>(capture_id);
#else
    (void)stream;
    throw std::runtime_error(
        "single-capture tensor debug probes require cudaStreamGetCaptureInfo");
#endif
}

void ProbeContext::validate_eager_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (captured_once_) {
        throw std::runtime_error(
            "tensor debug probe " + name_ +
            " has already been captured by a CUDA graph; eager calls after "
            "capture are unsupported");
    }
    if (!eager_stream_.has_value()) {
        eager_stream_ = stream;
        return;
    }
    if (*eager_stream_ != stream) {
        throw std::runtime_error(
            "tensor debug probe " + name_ +
            " already owns a different eager CUDA stream; create one probe "
            "per eager stream");
    }
}

uint64_t ProbeContext::next_slot_index(bool is_capturing, uint64_t capture_id) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (!is_capturing) {
        return 0;
    }

    if (!captured_once_) {
        captured_once_ = true;
        captured_capture_id_ = capture_id;
        next_slot_index_ = 0;
    } else if (captured_capture_id_ != capture_id) {
        std::ostringstream oss;
        oss << "tensor debug probe " << name_
            << " has already been captured by a CUDA graph; create a new probe "
               "for another capture";
        throw std::runtime_error(oss.str());
    }

    return next_slot_index_++;
}

torch::Tensor ProbeContext::source_tensor_for_enqueue(const torch::Tensor& tensor) const {
    if (tensor.is_contiguous()) {
        return tensor;
    }
    return tensor.contiguous();
}

TensorSlot& ProbeContext::ensure_slot(
    const torch::Tensor& tensor,
    size_t nbytes,
    uint64_t order,
    const std::string& observation_name,
    uint64_t invocation_index,
    bool is_capturing) {
    const size_t index = static_cast<size_t>(order);
    if (slots_.size() <= index) {
        slots_.resize(index + 1);
    }

    TensorSlot& slot = slots_[index];
    if (nbytes > slot.staging_nbytes) {
        void* new_staging = nullptr;
        {
            CaptureModeGuard guard(cudaStreamCaptureModeRelaxed);
            TCGD_CUDA_CHECK(cudaMallocHost(&new_staging, nbytes));
        }
        std::memset(new_staging, 0, nbytes);

        if (slot.staging != nullptr) {
            retired_staging_.push_back(slot.staging);
        }
        slot.staging = new_staging;
        slot.staging_nbytes = nbytes;
    }

    if (has_record_action_) {
        slot.observation.name = observation_name;
        slot.observation.replay_index = 0;
        slot.observation.order = order;
        slot.observation.invocation_index = invocation_index;
        slot.observation.shape = tensor.sizes().vec();
        slot.observation.dtype = tensor.scalar_type();
        slot.observation.device = tensor.device().str();
        slot.observation.nbytes = nbytes;
        slot.observation.valid = true;
        slot.observation.captured = is_capturing;
        slot.observation.bytes.clear();
    }
    return slot;
}

std::unique_ptr<CallbackPayload> ProbeContext::make_payload(
    const torch::Tensor& tensor,
    void* staging,
    size_t nbytes,
    uint64_t order,
    const std::string& observation_name,
    uint64_t invocation_index,
    bool is_capturing) {
    auto payload = std::make_unique<CallbackPayload>();
    payload->owner = this;
    payload->staging = staging;
    payload->nbytes = nbytes;
    payload->order = order;
    payload->name = observation_name;
    payload->invocation_index = invocation_index;
    payload->shape = tensor.sizes().vec();
    payload->dtype = tensor.scalar_type();
    payload->device = tensor.device().str();
    payload->numel = tensor.numel();
    payload->captured = is_capturing;
    return payload;
}

void ProbeContext::set_failure(
    uint64_t replay_index,
    int64_t order,
    std::string observation_name,
    int64_t invocation_index,
    const std::string& message) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (check_failed_) {
        return;
    }
    check_failed_ = true;
    failure_replay_index_ = replay_index;
    failure_order_ = order;
    failure_name_ = std::move(observation_name);
    failure_invocation_index_ = invocation_index;
    failure_message_ = message;
}

void ProbeContext::release_resources_noexcept() {
    if (replay_index_ready_event_ != nullptr) {
        cudaEventDestroy(replay_index_ready_event_);
        replay_index_ready_event_ = nullptr;
    }
    if (replay_index_staging_ != nullptr) {
        cudaFreeHost(replay_index_staging_);
        replay_index_staging_ = nullptr;
    }
    for (void* ptr : retired_staging_) {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
        }
    }
    retired_staging_.clear();
    for (TensorSlot& slot : slots_) {
        if (slot.staging != nullptr) {
            cudaFreeHost(slot.staging);
            slot.staging = nullptr;
            slot.staging_nbytes = 0;
        }
        slot.source_owners.clear();
    }
}

void CUDART_CB ProbeContext::host_callback(void* user_data) {
    auto* payload = static_cast<CallbackPayload*>(user_data);
    ProbeContext* owner = payload->owner;
    if (payload->captured) {
        owner->on_callback(*payload);
        return;
    }

    std::unique_ptr<CallbackPayload> owned(payload);
    owner->on_callback(*owned);
    owned.reset();
    owner->eager_callbacks_in_flight_.fetch_sub(1, std::memory_order_release);
}

std::shared_ptr<ProbeContext> make_probe_context(
    std::string name,
    std::vector<ActionConfig> actions,
    torch::Tensor replay_index,
    NonContiguousPolicy non_contiguous,
    ProbeMode mode) {
    const uint64_t id = next_context_id.fetch_add(1);
    auto context = std::make_shared<ProbeContext>(
        id,
        std::move(name),
        std::move(actions),
        std::move(replay_index),
        non_contiguous,
        mode);
    {
        std::lock_guard<std::mutex> guard(registry_mutex);
        registry.emplace(id, context);
    }
    return context;
}

}  // namespace torch_cudagraph_debug::tensor_debug
