#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "common/EventQueue.h"
#include "common/NetworkParser.h"
#include "congestion_aware/Chunk.h"
#include "congestion_aware/Helper.h"
#include "congestion_aware/Type.h"

namespace py = pybind11;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

int chunk_arrived_callback(void* const event_queue_ptr, void* const chunk_ptr) {
    // typecast event_queue_ptr
    auto* const event_queue = static_cast<EventQueue*>(event_queue_ptr);
    auto* const chunk = static_cast<Chunk*>(chunk_ptr);
    // Add the arrival request to the event queue and use it to clear out on the python side
    event_queue->add_arrival(chunk->get_req_id());
    // print chunk arrival time
    const auto current_time = event_queue->get_current_time();
    // std::cout << "A chunk arrived at destination at time: " << current_time << " ns" << "with id " << chunk->get_req_id() << std::endl;
    return 0;
}


PYBIND11_MODULE(simulation_py_congestion_aware, m) {
    py::class_<EventQueue, std::shared_ptr<EventQueue>>(m, "EventQueue")
        .def(py::init<>())
        .def("get_current_time", &EventQueue::get_current_time)
        .def("get_and_clear_arrivals", &EventQueue::get_and_clear_arrivals)
        .def("finished", &EventQueue::finished)
        .def("proceed", &EventQueue::proceed);

    py::class_<NetworkParser>(m, "NetworkParser")
        .def(py::init<const std::string&>());

    m.def("construct_topology", &construct_topology, py::arg("parser"));

    py::class_<Device, std::shared_ptr<Device>>(m, "Device");

    py::class_<Topology, std::shared_ptr<Topology>>(m, "Topology")
        .def("get_npus_count", &Topology::get_npus_count)
        .def("get_devices_count", &Topology::get_devices_count)
        .def("route", &Topology::route)
        .def("send_python", &Topology::send_python)
        .def_static("set_event_queue", &Topology::set_event_queue);

    py::class_<Chunk, std::shared_ptr<Chunk>>(m, "Chunk")
    .def_static("create_with_event_queue", [](ChunkSize chunk_size,
                                              int src_id,
                                              int dst_id,
                                              int req_id,
                                              Topology* topology,
                                              EventQueue* event_queue) {
        Route route = topology->route(src_id, dst_id);
        return std::make_shared<Chunk>(
            chunk_size,
            std::move(route),
            req_id,
            &chunk_arrived_callback,
            static_cast<void*>(event_queue)
        );
    }, py::arg("chunk_size"), py::arg("src_id"), py::arg("dst_id"),
       py::arg("req_id"), py::arg("topology"), py::arg("event_queue"));
}
