project (source)

add_library(${PROJECT_NAME} wolf.h math/tensor.cpp math/tensor.h 
model/Layer.h model/LinearLayer.h model/LinearLayer.cpp model/Sequential.h model/Sequential.cpp 
model/ReLU.h model/ReLU.cpp model/LayerFactory.h model/AdamStepper.cpp)

target_include_directories(source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal)


add_library(wolf::wolf ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC wolf_options)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(source PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/fp:fast>
)

set_source_files_properties(
    model/AdamStepper.cpp
    PROPERTIES
        COMPILE_OPTIONS
            "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-fast-math>;$<$<CXX_COMPILER_ID:MSVC>:/fp:precise>"
)
