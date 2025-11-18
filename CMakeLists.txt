cmake_minimum_required(VERSION 3.15)
set (CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O3 -ffast-math")
project(wolf)

find_package(TBB REQUIRED)

add_subdirectory(examples)
add_subdirectory(source)
add_subdirectory(test)