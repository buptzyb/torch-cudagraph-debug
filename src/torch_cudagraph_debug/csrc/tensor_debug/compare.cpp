#include "tensor_debug/compare.h"

#include "tensor_debug/tensor_format.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cmath>
#include <sstream>
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
CompareResult compare_typed(
    const void* actual_data,
    const void* expected_data,
    int64_t numel,
    double rtol,
    double atol,
    bool equal_nan) {
    const auto* actual = static_cast<const T*>(actual_data);
    const auto* expected = static_cast<const T*>(expected_data);

    for (int64_t i = 0; i < numel; ++i) {
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, c10::Half> ||
                      std::is_same_v<T, c10::BFloat16>) {
            const double a = to_double(actual[i]);
            const double e = to_double(expected[i]);
            if (a == e) {
                continue;
            }
            if (equal_nan && std::isnan(a) && std::isnan(e)) {
                continue;
            }
            const double abs_diff = std::abs(a - e);
            const double tolerance = atol + rtol * std::abs(e);
            if (abs_diff <= tolerance) {
                continue;
            }

            CompareResult result;
            result.ok = false;
            result.mismatch_index = i;
            result.actual = a;
            result.expected = e;
            result.abs_diff = abs_diff;
            result.tolerance = tolerance;
            std::ostringstream oss;
            oss << "mismatch at flattened index " << i << ": actual=" << a
                << " expected=" << e << " abs_diff=" << abs_diff
                << " tolerance=" << tolerance;
            result.message = oss.str();
            return result;
        } else {
            if (actual[i] == expected[i]) {
                continue;
            }

            CompareResult result;
            result.ok = false;
            result.mismatch_index = i;
            result.actual = to_double(actual[i]);
            result.expected = to_double(expected[i]);
            result.abs_diff = std::abs(result.actual - result.expected);
            result.tolerance = 0.0;
            std::ostringstream oss;
            oss << "mismatch at flattened index " << i << ": actual=" << result.actual
                << " expected=" << result.expected;
            result.message = oss.str();
            return result;
        }
    }

    return CompareResult{};
}

}  // namespace

CompareResult compare_tensor_bytes(
    const void* actual,
    const void* expected,
    int64_t numel,
    at::ScalarType dtype,
    double rtol,
    double atol,
    bool equal_nan) {
    switch (dtype) {
        case at::kFloat:
            return compare_typed<float>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kDouble:
            return compare_typed<double>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kHalf:
            return compare_typed<c10::Half>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kBFloat16:
            return compare_typed<c10::BFloat16>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kByte:
            return compare_typed<uint8_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kChar:
            return compare_typed<int8_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kShort:
            return compare_typed<int16_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kInt:
            return compare_typed<int32_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kLong:
            return compare_typed<int64_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kBool:
            return compare_typed<bool>(actual, expected, numel, rtol, atol, equal_nan);
        default:
            CompareResult result;
            result.ok = false;
            result.message = "unsupported dtype for compare: " + scalar_type_name(dtype);
            return result;
    }
}

}  // namespace torch_cudagraph_debug::tensor_debug
