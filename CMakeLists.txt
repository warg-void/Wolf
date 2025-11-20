cmake_minimum_required(VERSION 3.15)
project(wolf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(wolf_options INTERFACE)

target_compile_features(wolf_options INTERFACE cxx_std_23)

target_compile_options(wolf_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc>
)

find_package(OpenMP REQUIRED)

add_subdirectory(examples)
add_subdirectory(source)
