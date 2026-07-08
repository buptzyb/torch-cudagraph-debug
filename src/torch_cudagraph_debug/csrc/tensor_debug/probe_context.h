#pragma once

#include <torch/extension.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
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
    std::string observation_name;
    uint64_t invocation_index = 0;
    std::vector<uint8_t> expected_bytes;
    std::vector<int64_t> expected_shape;
    at::ScalarType expected_dtype = at::kFloat;
    int64_t expected_numel = 0;
};

struct CheckActionConfig {
    bool keyed = false;
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
    std::string name;
    uint64_t replay_index = 0;
    uint64_t order = 0;
    uint64_t invocation_index = 0;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kFloat;
    std::string device;
    std::vector<uint8_t> bytes;
    size_t nbytes = 0;
    bool valid = false;
    bool captured = false;
    // Times an eager slot was re-sampled in place; lets consumers tell a
    // latest-value sample from a complete history.
    uint64_t eager_overwrites = 0;
};

struct TensorSlot {
    void* staging = nullptr;
    size_t staging_nbytes = 0;
    TensorObservationData observation;
    std::vector<torch::Tensor> source_owners;
};

struct CallbackPayload {
    class ProbeContext* owner = nullptr;
    void* staging = nullptr;
    size_t nbytes = 0;
    uint64_t order = 0;
    std::string name;
    uint64_t invocation_index = 0;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kFloat;
    std::string device;
    int64_t numel = 0;
    bool captured = false;
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

    torch::Tensor enqueue(
        const torch::Tensor& tensor,
        const std::string& observation_name,
        uint64_t invocation_index);
    pybind11::list observations(std::optional<uint64_t> replay_index);
    pybind11::dict check_status();
    pybind11::dict debug_resource_counts() const;
    void debug_fail_next_enqueue(const std::string& stage);
    void reclaim_retired_staging();
    void close();

    void on_callback(const CallbackPayload& payload) noexcept;
    uint64_t id() const { return id_; }

  private:
    enum class DebugFailureStage {
        None,
        HostPreparation,
        AfterCudaSubmission,
    };

    struct SlotCommit {
        bool is_capturing = false;
        bool inserted_eager_slot = false;
        bool updating_eager_slot = false;
        bool claimed_eager_stream = false;
        bool began_capture = false;
        uint64_t order = 0;
        std::string observation_name;
        void* prior_staging = nullptr;
        size_t prior_staging_nbytes = 0;
        size_t prior_retired_staging_size = 0;
        TensorObservationData prior_observation;
        std::vector<TensorSlot> prior_eager_slots;
        std::unordered_map<std::string, uint64_t> prior_eager_slot_by_name;
    };

    void ensure_open() const;
    void validate_tensor(const torch::Tensor& tensor) const;
    void validate_check_actions(
        const torch::Tensor& tensor,
        uint64_t order,
        const std::string& observation_name,
        uint64_t invocation_index) const;
    uint64_t capture_id_for_stream(cudaStream_t stream) const;
    void validate_eager_stream(cudaStream_t stream);
    uint64_t peek_slot_index(
        bool is_capturing,
        uint64_t capture_id,
        const std::string& observation_name) const;
    SlotCommit commit_slot_index(
        bool is_capturing,
        uint64_t capture_id,
        uint64_t order,
        const std::string& observation_name,
        cudaStream_t stream);
    void finalize_slot_commit(SlotCommit& commit) noexcept;
    bool rollback_slot_commit(SlotCommit& commit) noexcept;
    void maybe_fail_debug_enqueue(DebugFailureStage stage);
    void mark_poisoned(const char* operation) noexcept;
    torch::Tensor source_tensor_for_enqueue(const torch::Tensor& tensor) const;
    TensorSlot& ensure_slot(
        const torch::Tensor& tensor,
        size_t nbytes,
        uint64_t order,
        const std::string& observation_name,
        uint64_t invocation_index,
        bool is_capturing);
    std::unique_ptr<CallbackPayload> make_payload(
        const torch::Tensor& tensor,
        void* staging,
        size_t nbytes,
        uint64_t order,
        const std::string& observation_name,
        uint64_t invocation_index,
        bool is_capturing);
    void set_failure(
        uint64_t replay_index,
        int64_t order,
        std::string observation_name,
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
    cudaEvent_t eager_work_event_ = nullptr;
    NonContiguousPolicy non_contiguous_;
    ProbeMode mode_;
    bool has_record_action_ = false;
    bool has_callback_actions_ = false;

    std::vector<void*> retired_staging_;
    std::vector<std::unique_ptr<CallbackPayload>> captured_payloads_;
    std::vector<TensorSlot> slots_;
    std::atomic<uint64_t> eager_callbacks_in_flight_{0};

    mutable std::mutex enqueue_mutex_;
    mutable std::mutex mutex_;
    bool closed_ = false;
    bool poisoned_ = false;
    const char* poison_operation_ = "enqueue";
    DebugFailureStage debug_failure_stage_ = DebugFailureStage::None;

    bool captured_once_ = false;
    uint64_t captured_capture_id_ = 0;
    uint64_t next_slot_index_ = 0;
    std::unordered_map<std::string, uint64_t> eager_slot_by_name_;
    uint64_t eager_callback_count_ = 0;
    std::optional<cudaStream_t> eager_stream_;

    bool check_failed_ = false;
    uint64_t failure_replay_index_ = 0;
    int64_t failure_order_ = -1;
    std::string failure_name_;
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
