#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace torch_cudagraph_debug::tensor_debug {

void launch_increment_replay_counter(int64_t* counter, cudaStream_t stream);

}  // namespace torch_cudagraph_debug::tensor_debug
