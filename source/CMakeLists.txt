project (source)

add_library(${PROJECT_NAME} wolf.h math/tensor.cpp math/tensor.h 
model/Layer.h model/LinearLayer.h model/LinearLayer.cpp model/Sequential.h model/Sequential.cpp 
model/ReLU.h model/ReLU.cpp model/LayerFactory.h math/loss.h)

target_include_directories(source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal)


add_library(wolf::wolf ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
