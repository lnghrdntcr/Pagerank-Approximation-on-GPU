# cmake_minimum_required(VERSION <specify CMake version here>)
cmake_minimum_required(VERSION 3.14)
project(Approximate_PR)

set(CMAKE_CUDA_COMPILER  /opt/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 17)
enable_language("CUDA")

add_executable(Approximate_PR np_main.cu main.cu Parse/Parse.h Parse/Parse.cpp Matrix/Matrix.cpp Matrix/Matrix.h Utils/Utils.cpp Utils/Utils.h)