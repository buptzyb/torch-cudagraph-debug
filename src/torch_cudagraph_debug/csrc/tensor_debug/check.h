#pragma once

#include <ATen/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace torch_cudagraph_debug::tensor_debug {

struct CheckResult {
    bool ok = true;
    int64_t mismatch_index = -1;
    std::string message;
};

CheckResult check_tensor_bytes(
    const void* actual,
    const void* expected,
    int64_t numel,
    at::ScalarType dtype,
    double rtol,
    double atol,
    bool equal_nan);

}  // namespace torch_cudagraph_debug::tensor_debug
