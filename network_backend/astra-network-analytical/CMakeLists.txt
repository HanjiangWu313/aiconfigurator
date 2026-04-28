# CMake Requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Setup project
project(Analytical)

# Auto-discover pybind11 from the active Python environment
find_program(_PYTHON3 python3)
execute_process(
    COMMAND "${_PYTHON3}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_result
)
if (_pybind11_result EQUAL 0 AND _pybind11_cmake_dir)
    set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake dir (auto-detected)" FORCE)
endif ()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_unaware/congestion_aware)")

# Can be compiled into either library or executable
option(NETWORK_BACKEND_BUILD_AS_LIBRARY "Build as a library" ON)

# Compile external libraries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/yaml-cpp yaml-cpp)

# Include src files to compile
file(GLOB srcs_common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/event-queue/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/network-parser/*.cpp
)

file(GLOB srcs_congestion_unaware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/topology/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/basic-topology/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/multi-dim-topology/*.cpp
)

file(GLOB srcs_congestion_aware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/network/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/topology/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/basic-topology/*.cpp
)

# Compile Congestion Unaware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_unaware")
    if (NETWORK_BACKEND_BUILD_AS_LIBRARY)
        add_library(Analytical_Congestion_Unaware STATIC ${srcs_congestion_unaware} ${srcs_common})

        # Properties
        set_target_properties(Analytical_Congestion_Unaware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        )
    else ()
        add_executable(Analytical_Congestion_Unaware ${srcs_congestion_unaware} ${srcs_common})
        target_sources(Analytical_Congestion_Unaware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/example.cpp)

        # Properties
        set_target_properties(Analytical_Congestion_Unaware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
        )
    endif ()

    # Common properties
    set_target_properties(Analytical_Congestion_Unaware PROPERTIES COMPILE_WARNING_AS_ERROR ON)

    # Link libraries
    target_link_libraries(Analytical_Congestion_Unaware PUBLIC yaml-cpp)

    # Include directories
    target_include_directories(Analytical_Congestion_Unaware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(Analytical_Congestion_Unaware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/)
    target_include_directories(Analytical_Congestion_Unaware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/)

    set_target_properties(Analytical_Congestion_Unaware PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Create Python bindings for Congestion-unaware Backend
    pybind11_add_module(simulation_py_congestion_unaware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/simulation_pybindings.cpp
    )

    target_link_libraries(simulation_py_congestion_unaware
        PRIVATE
            Analytical_Congestion_Unaware
            yaml-cpp
    )

    target_include_directories(simulation_py_congestion_unaware
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include/
            ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/
    )

    set_target_properties(simulation_py_congestion_unaware PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
endif ()

# Compile Congestion Aware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_aware")
    if (NETWORK_BACKEND_BUILD_AS_LIBRARY)
        add_library(Analytical_Congestion_Aware STATIC ${srcs_congestion_aware} ${srcs_common})

        # Properties
        set_target_properties(Analytical_Congestion_Aware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        )
    else ()
        add_executable(Analytical_Congestion_Aware ${srcs_congestion_aware} ${srcs_common})
        target_sources(Analytical_Congestion_Aware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/example.cpp)

        # Properties
        set_target_properties(Analytical_Congestion_Aware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
        )
    endif ()

    # Common properties
    set_target_properties(Analytical_Congestion_Aware PROPERTIES COMPILE_WARNING_AS_ERROR ON)

    # Link libraries
    target_link_libraries(Analytical_Congestion_Aware PUBLIC yaml-cpp)

    # Include directories
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/)
    target_include_directories(Analytical_Congestion_Aware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/)
    
    set_target_properties(Analytical_Congestion_Aware PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Create Python bindings for Congestion aware Backend

    # Define pybind11 module target
    pybind11_add_module(simulation_py_congestion_aware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/simulation_pybindings.cpp
    )

    # Link against your congestion-aware library
    target_link_libraries(simulation_py_congestion_aware
        PRIVATE
            Analytical_Congestion_Aware
            yaml-cpp
    )

    # Include your headers for pybind to see your C++ classes
    target_include_directories(simulation_py_congestion_aware
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include/
            ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/
    )

    # Optional: Set output directory for Python module
    set_target_properties(simulation_py_congestion_aware PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
endif ()



