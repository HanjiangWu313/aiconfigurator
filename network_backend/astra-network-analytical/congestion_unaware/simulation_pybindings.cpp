/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common/NetworkParser.h"
#include "congestion_unaware/Helper.h"
#include "congestion_unaware/Topology.h"

namespace py = pybind11;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;


PYBIND11_MODULE(simulation_py_congestion_unaware, m) {
    m.doc() = "Congestion-unaware analytical network simulator with multi-dim topology support";

    // NetworkParser is shared with the congestion_aware module.
    // Use module_local() to avoid global type-registration conflicts
    // when both modules are loaded in the same Python interpreter.
    py::class_<NetworkParser>(m, "NetworkParser", py::module_local())
        .def(py::init<const std::string&>())
        .def("get_dims_count", &NetworkParser::get_dims_count)
        .def("get_npus_counts_per_dim", &NetworkParser::get_npus_counts_per_dim)
        .def("get_bandwidths_per_dim", &NetworkParser::get_bandwidths_per_dim)
        .def("get_latencies_per_dim", &NetworkParser::get_latencies_per_dim);

    py::class_<Topology, std::shared_ptr<Topology>>(m, "Topology", py::module_local())
        .def("send", &Topology::send,
             py::arg("src"), py::arg("dest"), py::arg("chunk_size"),
             "Estimate transfer time (ns) from src to dest for chunk_size bytes.\n\n"
             "For multi-dim topologies, routing is automatic: the address is\n"
             "decomposed and the transfer uses the correct dimension's bandwidth.")
        .def("get_npus_count", &Topology::get_npus_count)
        .def("get_dims_count", &Topology::get_dims_count)
        .def("get_npus_count_per_dim", &Topology::get_npus_count_per_dim)
        .def("get_bandwidth_per_dim", &Topology::get_bandwidth_per_dim);

    m.def("construct_topology", &construct_topology, py::arg("parser"),
          "Build a (possibly multi-dim) topology from a NetworkParser.\n\n"
          "If the YAML config has a single dimension a BasicTopology is\n"
          "returned; otherwise a MultiDimTopology is constructed.");
}
