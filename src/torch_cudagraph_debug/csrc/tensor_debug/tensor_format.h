#pragma once

#include <ATen/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace torch_cudagraph_debug::tensor_debug {

std::string scalar_type_name(at::ScalarType dtype);

size_t scalar_type_size(at::ScalarType dtype);

std::string format_tensor_bytes(
    const void* data,
    int64_t numel,
    at::ScalarType dtype,
    const std::vector<int64_t>& shape,
    int64_t max_items,
    bool include_summary);

}  // namespace torch_cudagraph_debug::tensor_debug
