cmake_minimum_required(VERSION 3.14)
project(toy_bindings)

set(CMAKE_CXX_STANDARD 17)

# include_directories("/home/hanjiang/miniconda3/envs/GenA/bin/python")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
# add_subdirectory(pybind11)  # Make sure pybind11 is cloned here

pybind11_add_module(toy_network_backend toy_network_backend.cpp)