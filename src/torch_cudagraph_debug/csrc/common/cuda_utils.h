#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace torch_cudagraph_debug {

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(status) << " (" << static_cast<int>(status) << ")";
    throw std::runtime_error(oss.str());
}

#define TCGD_CUDA_CHECK(expr) \
    ::torch_cudagraph_debug::cuda_check((expr), #expr, __FILE__, __LINE__)

class CaptureModeGuard {
  public:
    explicit CaptureModeGuard(cudaStreamCaptureMode desired) : previous_(desired), active_(false) {
        TCGD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&previous_));
        active_ = true;
    }

    CaptureModeGuard(const CaptureModeGuard&) = delete;
    CaptureModeGuard& operator=(const CaptureModeGuard&) = delete;

    ~CaptureModeGuard() {
        if (active_) {
            cudaThreadExchangeStreamCaptureMode(&previous_);
        }
    }

  private:
    cudaStreamCaptureMode previous_;
    bool active_;
};

}  // namespace torch_cudagraph_debug
