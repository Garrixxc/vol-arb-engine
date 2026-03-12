cmake_minimum_required(VERSION 3.15)
project(vol_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with aggressive optimization
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -march=native -ffast-math)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(vol_core
    black_scholes.cpp
    iv_solver.cpp
    svi_calibrator.cpp
    bindings.cpp
)

target_include_directories(vol_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
