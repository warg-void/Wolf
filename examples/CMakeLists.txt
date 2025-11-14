project(examples)

add_executable(singleNeuron singleNeuron.cpp)
target_link_libraries(singleNeuron wolf::wolf)
