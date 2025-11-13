project(examples)

add_executable(Example1 main.cpp)
target_link_libraries(Example1 wolf::wolf)
