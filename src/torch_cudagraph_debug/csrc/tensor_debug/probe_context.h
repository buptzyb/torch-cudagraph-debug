#pragma once

#include <torch/extension.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace torch_cudagraph_debug::tensor_debug {

struct PrintActionConfig {
    bool enabled = true;
    int64_t max_items = 16;
    int64_t every = 1;
    bool summary = true;
};

struct RecordActionConfig {
    bool enabled = true;
};

struct ExpectedTensorConfig {
    std::vector<uint8_t> expected_bytes;
    std::vector<int64_t> expected_shape;
    at::ScalarType expected_dtype = at::kFloat;
    int64_t expected_numel = 0;
};

struct CheckActionConfig {
    bool enabled = true;
    std::vector<ExpectedTensorConfig> expected;
    double rtol = 1e-5;
    double atol = 1e-8;
    bool equal_nan = false;
};

struct ActionConfig {
    enum class Kind { Print, Record, Check };

    Kind kind;
    PrintActionConfig print;
    RecordActionConfig record;
    CheckActionConfig check;
};

enum class NonContiguousPolicy { Error, Copy };

enum class ProbeMode { Capture, Always };

struct TensorObservationData {
    std::string probe_name;
    uint64_t replay_index = 0;
    uint64_t invocation_index = 0;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kFloat;
    std::string device;
    std::vector<uint8_t> bytes;
    size_t nbytes = 0;
    bool valid = false;
    bool captured = false;
};

struct InvocationSlot {
    void* staging = nullptr;
    size_t staging_nbytes = 0;
    TensorObservationData observation;
    std::vector<torch::Tensor> source_owners;
};

struct CallbackPayload {
    class ProbeContext* owner = nullptr;
    void* staging = nullptr;
    size_t nbytes = 0;
    uint64_t invocation_index = 0;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kFloat;
    std::string device;
    int64_t numel = 0;
    bool captured = false;
    torch::Tensor source_owner;
};

class ProbeContext {
  public:
    ProbeContext(
        uint64_t id,
        std::string name,
        std::vector<ActionConfig> actions,
        torch::Tensor replay_index,
        NonContiguousPolicy non_contiguous,
        ProbeMode mode);
    ~ProbeContext();

    ProbeContext(const ProbeContext&) = delete;
    ProbeContext& operator=(const ProbeContext&) = delete;

    torch::Tensor enqueue(const torch::Tensor& tensor);
    pybind11::list observations(std::optional<uint64_t> replay_index);
    void clear_observations();
    pybind11::dict check_status();
    void close();

    void on_callback(const CallbackPayload& payload) noexcept;
    uint64_t id() const { return id_; }

  private:
    void ensure_open() const;
    void validate_tensor(const torch::Tensor& tensor) const;
    void validate_check_actions(
        const torch::Tensor& tensor,
        uint64_t invocation_index) const;
    uint64_t capture_id_for_stream(cudaStream_t stream) const;
    uint64_t next_invocation_index(bool is_capturing, uint64_t capture_id);
    torch::Tensor source_tensor_for_enqueue(const torch::Tensor& tensor) const;
    InvocationSlot& ensure_invocation_slot(
        const torch::Tensor& tensor,
        size_t nbytes,
        uint64_t invocation_index,
        bool is_capturing);
    CallbackPayload* add_payload(
        const torch::Tensor& tensor,
        const torch::Tensor& source,
        void* staging,
        size_t nbytes,
        uint64_t invocation_index,
        bool is_capturing);
    void set_failure(
        uint64_t replay_index,
        int64_t invocation_index,
        const std::string& message);
    void release_resources_noexcept();

    static void CUDART_CB host_callback(void* user_data);

    uint64_t id_;
    std::string name_;
    std::vector<ActionConfig> actions_;
    torch::Tensor replay_index_;
    int replay_index_device_ = -1;
    int64_t* replay_index_staging_ = nullptr;
    cudaEvent_t replay_index_ready_event_ = nullptr;
    NonContiguousPolicy non_contiguous_;
    ProbeMode mode_;
    bool has_record_action_ = false;
    bool has_callback_actions_ = false;

    std::vector<void*> retired_staging_;
    std::vector<std::unique_ptr<CallbackPayload>> payloads_;
    std::vector<InvocationSlot> invocation_slots_;

    mutable std::mutex mutex_;
    bool closed_ = false;

    bool captured_once_ = false;
    uint64_t captured_capture_id_ = 0;
    uint64_t next_invocation_index_ = 0;
    uint64_t eager_callback_count_ = 0;

    bool check_failed_ = false;
    uint64_t failure_replay_index_ = 0;
    int64_t failure_invocation_index_ = -1;
    std::string failure_message_;
};

std::shared_ptr<ProbeContext> make_probe_context(
    std::string name,
    std::vector<ActionConfig> actions,
    torch::Tensor replay_index,
    NonContiguousPolicy non_contiguous,
    ProbeMode mode);

}  // namespace torch_cudagraph_debug::tensor_debug
