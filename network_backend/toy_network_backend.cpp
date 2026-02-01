#include <pybind11/pybind11.h>

namespace py = pybind11;

int run_astra_sim_step(int req_step_start_time) {
    // TODO: Run AstraSim
    int current_time = req_step_start_time + 10;
    return current_time;
}

PYBIND11_MODULE(toy_network_backend, m) {
    m.def("run_astra_sim_step", &run_astra_sim_step, "Run one Astra-Sim step");
}