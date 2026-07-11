#include "tensor_debug/replay_counter.h"

#include "common/cuda_utils.h"

namespace torch_cudagraph_debug::tensor_debug {
namespace {

__global__ void increment_replay_counter(int64_t* counter) {
    *counter += 1;
}

}  // namespace

void launch_increment_replay_counter(int64_t* counter, cudaStream_t stream) {
    increment_replay_counter<<<1, 1, 0, stream>>>(counter);
    TCGD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace torch_cudagraph_debug::tensor_debug
