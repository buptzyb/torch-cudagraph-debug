#include "tensor_debug/check.h"

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
uint64_t integer_abs_difference(T actual, T expected) {
    if (actual >= expected) {
        return static_cast<uint64_t>(actual) - static_cast<uint64_t>(expected);
    }
    return static_cast<uint64_t>(expected) - static_cast<uint64_t>(actual);
}

template <typename T>
void append_integer(std::ostringstream& stream, T value) {
    if constexpr (std::is_same_v<T, bool>) {
        stream << (value ? 1 : 0);
    } else if constexpr (std::is_signed_v<T>) {
        stream << static_cast<int64_t>(value);
    } else {
        stream << static_cast<uint64_t>(value);
    }
}

template <typename T>
CheckResult check_typed(
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
            if (std::isfinite(a) && std::isfinite(e) && abs_diff <= tolerance) {
                continue;
            }

            CheckResult result;
            result.ok = false;
            result.mismatch_index = i;
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

            CheckResult result;
            result.ok = false;
            result.mismatch_index = i;
            const uint64_t abs_diff = integer_abs_difference(actual[i], expected[i]);
            std::ostringstream oss;
            oss << "mismatch at flattened index " << i << ": actual=";
            append_integer(oss, actual[i]);
            oss << " expected=";
            append_integer(oss, expected[i]);
            oss << " abs_diff=" << abs_diff;
            result.message = oss.str();
            return result;
        }
    }

    return CheckResult{};
}

}  // namespace

CheckResult check_tensor_bytes(
    const void* actual,
    const void* expected,
    int64_t numel,
    at::ScalarType dtype,
    double rtol,
    double atol,
    bool equal_nan) {
    switch (dtype) {
        case at::kFloat:
            return check_typed<float>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kDouble:
            return check_typed<double>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kHalf:
            return check_typed<c10::Half>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kBFloat16:
            return check_typed<c10::BFloat16>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kByte:
            return check_typed<uint8_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kChar:
            return check_typed<int8_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kShort:
            return check_typed<int16_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kInt:
            return check_typed<int32_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kLong:
            return check_typed<int64_t>(actual, expected, numel, rtol, atol, equal_nan);
        case at::kBool:
            return check_typed<bool>(actual, expected, numel, rtol, atol, equal_nan);
        default:
            CheckResult result;
            result.ok = false;
            result.message = "unsupported dtype for check: " + scalar_type_name(dtype);
            return result;
    }
}

}  // namespace torch_cudagraph_debug::tensor_debug
