#include "tensor_debug/probe_context.h"

#include <torch/extension.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace torch_cudagraph_debug::tensor_debug {
namespace {

bool get_bool(py::dict dict, const char* key, bool default_value) {
    if (!dict.contains(key)) {
        return default_value;
    }
    return py::cast<bool>(dict[key]);
}

int64_t get_int64(py::dict dict, const char* key, int64_t default_value) {
    if (!dict.contains(key)) {
        return default_value;
    }
    return py::cast<int64_t>(dict[key]);
}

double get_double(py::dict dict, const char* key, double default_value) {
    if (!dict.contains(key)) {
        return default_value;
    }
    return py::cast<double>(dict[key]);
}

ExpectedTensorConfig parse_expected_tensor(torch::Tensor expected, const char* action_name) {
    if (!expected.defined()) {
        throw std::runtime_error(std::string(action_name) + " expected tensor must be defined");
    }
    if (expected.is_cuda()) {
        throw std::runtime_error(std::string(action_name) + " expected tensor must be a CPU tensor");
    }

    expected = expected.contiguous();

    ExpectedTensorConfig result;
    result.expected_shape = expected.sizes().vec();
    result.expected_dtype = expected.scalar_type();
    result.expected_numel = expected.numel();

    const size_t nbytes = static_cast<size_t>(expected.numel() * expected.element_size());
    result.expected_bytes.resize(nbytes);
    if (nbytes > 0) {
        std::memcpy(result.expected_bytes.data(), expected.data_ptr(), nbytes);
    }
    return result;
}

std::vector<ActionConfig> parse_actions(py::list action_specs) {
    std::vector<ActionConfig> actions;
    actions.reserve(py::len(action_specs));

    for (py::handle item : action_specs) {
        py::dict spec = py::cast<py::dict>(item);
        const std::string kind = py::cast<std::string>(spec["kind"]);

        ActionConfig action;
        if (kind == "print") {
            action.kind = ActionConfig::Kind::Print;
            action.print.enabled = get_bool(spec, "enabled", true);
            action.print.max_items = get_int64(spec, "max_items", 16);
            action.print.every = get_int64(spec, "every", 1);
            action.print.summary = get_bool(spec, "summary", true);
        } else if (kind == "record") {
            action.kind = ActionConfig::Kind::Record;
            action.record.enabled = get_bool(spec, "enabled", true);
        } else if (kind == "check") {
            action.kind = ActionConfig::Kind::Check;
            action.check.enabled = get_bool(spec, "enabled", true);
            action.check.rtol = get_double(spec, "rtol", 1e-5);
            action.check.atol = get_double(spec, "atol", 1e-8);
            action.check.equal_nan = get_bool(spec, "equal_nan", false);

            py::list expected_items = py::cast<py::list>(spec["expected"]);
            if (py::len(expected_items) == 0) {
                throw std::runtime_error("CheckAction expected list must be non-empty");
            }
            action.check.expected.reserve(py::len(expected_items));
            for (py::handle expected_item : expected_items) {
                torch::Tensor expected = py::cast<torch::Tensor>(expected_item);
                action.check.expected.push_back(
                    parse_expected_tensor(expected, "CheckAction"));
            }
        } else {
            throw std::runtime_error("unknown tensor debug action kind: " + kind);
        }

        actions.push_back(std::move(action));
    }

    if (actions.empty()) {
        throw std::runtime_error("at least one tensor debug action is required");
    }
    return actions;
}

NonContiguousPolicy parse_non_contiguous_policy(const std::string& policy) {
    if (policy == "error") {
        return NonContiguousPolicy::Error;
    }
    if (policy == "copy") {
        return NonContiguousPolicy::Copy;
    }
    throw std::runtime_error("non_contiguous must be either \"error\" or \"copy\"");
}

ProbeMode parse_probe_mode(const std::string& mode) {
    if (mode == "capture") {
        return ProbeMode::Capture;
    }
    if (mode == "always") {
        return ProbeMode::Always;
    }
    throw std::runtime_error("mode must be either \"capture\" or \"always\"");
}

std::shared_ptr<ProbeContext> create_tensor_debug_probe(
    const std::string& name,
    py::list action_specs,
    torch::Tensor replay_index,
    const std::string& non_contiguous,
    const std::string& mode) {
    if (name.empty()) {
        throw std::runtime_error("probe name must be non-empty");
    }
    return make_probe_context(
        name,
        parse_actions(action_specs),
        std::move(replay_index),
        parse_non_contiguous_policy(non_contiguous),
        parse_probe_mode(mode));
}

}  // namespace
}  // namespace torch_cudagraph_debug::tensor_debug

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<
        torch_cudagraph_debug::tensor_debug::ProbeContext,
        std::shared_ptr<torch_cudagraph_debug::tensor_debug::ProbeContext>>(
        m, "TensorDebugProbeHandle")
        .def(
            "enqueue",
            &torch_cudagraph_debug::tensor_debug::ProbeContext::enqueue,
            py::arg("tensor"),
            py::arg("name"),
            py::arg("invocation_index"))
        .def(
            "observations",
            &torch_cudagraph_debug::tensor_debug::ProbeContext::observations,
            py::arg("replay_index") = py::none())
        .def("clear_observations", &torch_cudagraph_debug::tensor_debug::ProbeContext::clear_observations)
        .def("check_status", &torch_cudagraph_debug::tensor_debug::ProbeContext::check_status)
        .def(
            "_debug_resource_counts",
            &torch_cudagraph_debug::tensor_debug::ProbeContext::debug_resource_counts)
        .def(
            "_reclaim_retired_staging",
            &torch_cudagraph_debug::tensor_debug::ProbeContext::reclaim_retired_staging)
        .def("close", &torch_cudagraph_debug::tensor_debug::ProbeContext::close);

    m.def(
        "create_tensor_debug_probe",
        &torch_cudagraph_debug::tensor_debug::create_tensor_debug_probe,
        py::arg("name"),
        py::arg("actions"),
        py::arg("replay_index"),
        py::arg("non_contiguous") = "error",
        py::arg("mode") = "capture");
}
