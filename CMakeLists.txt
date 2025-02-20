cmake_minimum_required(VERSION 3.29)
project(DPF)
include_directories(./lib/eigen)  # Sous Windows (décommente cette ligne si nécessaire)

set(CMAKE_CXX_STANDARD 20)

add_executable(DPF main.cpp)
