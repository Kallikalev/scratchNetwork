cmake_minimum_required(VERSION 3.23)
project(scratchNetwork)

set(CMAKE_CXX_STANDARD 14)

add_executable(scratchNetwork
        src/DenseLayer.cpp
        src/main.cpp
        src/Network.cpp
        src/TanhLayer.cpp)
