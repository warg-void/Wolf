project (source)
add_library(${PROJECT_NAME} wolf.h math/tensor.cpp math/tensor.h 
model/Layer.h model/LinearLayer.h model/LinearLayer.cpp model/Sequential.h model/Sequential.cpp 
model/ReLU.h model/ReLU.cpp model/LayerFactory.h math/loss.h)

target_include_directories(source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/math)


add_library(wolf::wolf ALIAS ${PROJECT_NAME})
