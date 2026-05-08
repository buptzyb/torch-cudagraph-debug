#include "tensor_debug/tensor_format.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace torch_cudagraph_debug::tensor_debug {
namespace {

template <typename T>
double to_double(T value) {
    if constexpr (std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16>) {
        return static_cast<float>(value);
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? 1.0 : 0.0;
    } else {
        return static_cast<double>(value);
    }
}

template <typename T>
std::string value_to_string(T value) {
    std::ostringstream oss;
    if constexpr (std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16>) {
        oss << static_cast<float>(value);
    } else if constexpr (std::is_same_v<T, bool>) {
        oss << (value ? "true" : "false");
    } else if constexpr (std::is_integral_v<T> && sizeof(T) == 1 && !std::is_same_v<T, bool>) {
        oss << static_cast<int64_t>(value);
    } else {
        oss << value;
    }
    return oss.str();
}

template <typename T>
std::string format_typed(
    const void* data,
    int64_t numel,
    const std::vector<int64_t>& shape,
    int64_t max_items,
    bool include_summary) {
    const auto* values = static_cast<const T*>(data);
    std::ostringstream oss;
    oss << "shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";

    if (include_summary) {
        double min_value = std::numeric_limits<double>::infinity();
        double max_value = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        int64_t finite_count = 0;
        int64_t nan_count = 0;
        int64_t inf_count = 0;

        for (int64_t i = 0; i < numel; ++i) {
            const double value = to_double(values[i]);
            if (std::isnan(value)) {
                ++nan_count;
                continue;
            }
            if (std::isinf(value)) {
                ++inf_count;
            } else {
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
                sum += value;
                ++finite_count;
            }
        }

        oss << " numel=" << numel;
        if (finite_count > 0) {
            oss << " min=" << min_value << " max=" << max_value
                << " mean=" << (sum / static_cast<double>(finite_count));
        }
        oss << " nan=" << nan_count << " inf=" << inf_count;
    }

    const int64_t sample_count = std::min<int64_t>(std::max<int64_t>(max_items, 0), numel);
    oss << " values=[";
    for (int64_t i = 0; i < sample_count; ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << value_to_string(values[i]);
    }
    if (sample_count < numel) {
        if (sample_count > 0) {
            oss << ", ";
        }
        oss << "...";
    }
    oss << "]";
    return oss.str();
}

}  // namespace

std::string scalar_type_name(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return "float32";
        case at::kDouble:
            return "float64";
        case at::kHalf:
            return "float16";
        case at::kBFloat16:
            return "bfloat16";
        case at::kByte:
            return "uint8";
        case at::kChar:
            return "int8";
        case at::kShort:
            return "int16";
        case at::kInt:
            return "int32";
        case at::kLong:
            return "int64";
        case at::kBool:
            return "bool";
        default:
            return "unsupported";
    }
}

size_t scalar_type_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return sizeof(float);
        case at::kDouble:
            return sizeof(double);
        case at::kHalf:
            return sizeof(c10::Half);
        case at::kBFloat16:
            return sizeof(c10::BFloat16);
        case at::kByte:
            return sizeof(uint8_t);
        case at::kChar:
            return sizeof(int8_t);
        case at::kShort:
            return sizeof(int16_t);
        case at::kInt:
            return sizeof(int32_t);
        case at::kLong:
            return sizeof(int64_t);
        case at::kBool:
            return sizeof(bool);
        default:
            throw std::runtime_error("unsupported dtype: " + scalar_type_name(dtype));
    }
}

std::string format_tensor_bytes(
    const void* data,
    int64_t numel,
    at::ScalarType dtype,
    const std::vector<int64_t>& shape,
    int64_t max_items,
    bool include_summary) {
    switch (dtype) {
        case at::kFloat:
            return format_typed<float>(data, numel, shape, max_items, include_summary);
        case at::kDouble:
            return format_typed<double>(data, numel, shape, max_items, include_summary);
        case at::kHalf:
            return format_typed<c10::Half>(data, numel, shape, max_items, include_summary);
        case at::kBFloat16:
            return format_typed<c10::BFloat16>(data, numel, shape, max_items, include_summary);
        case at::kByte:
            return format_typed<uint8_t>(data, numel, shape, max_items, include_summary);
        case at::kChar:
            return format_typed<int8_t>(data, numel, shape, max_items, include_summary);
        case at::kShort:
            return format_typed<int16_t>(data, numel, shape, max_items, include_summary);
        case at::kInt:
            return format_typed<int32_t>(data, numel, shape, max_items, include_summary);
        case at::kLong:
            return format_typed<int64_t>(data, numel, shape, max_items, include_summary);
        case at::kBool:
            return format_typed<bool>(data, numel, shape, max_items, include_summary);
        default:
            throw std::runtime_error("unsupported dtype: " + scalar_type_name(dtype));
    }
}

}  // namespace torch_cudagraph_debug::tensor_debug
