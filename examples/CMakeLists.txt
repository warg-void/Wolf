project(examples)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)

# Copy iris.csv from source tree to build tree
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/iris.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

add_executable(singleNeuron singleNeuron.cpp)
add_executable(xortest xortest.cpp)
add_executable(irisClassifer irisClassifier.cpp)

target_link_libraries(singleNeuron PRIVATE wolf::wolf wolf_options)
target_link_libraries(xortest PRIVATE wolf::wolf wolf_options)
target_link_libraries(irisClassifer PRIVATE wolf::wolf wolf_options)
option(BUILD_MNIST "Build MNIST example" ON)
if (BUILD_MNIST)
    add_executable(mnistClassifier mnistClassifier.cpp)
    target_link_libraries(mnistClassifier PRIVATE wolf::wolf wolf_options)

    set(MNIST_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
    set(MNIST_ZIP      ${MNIST_DATA_DIR}/mnist-in-csv.zip)
    set(MNIST_TRAIN_CSV ${MNIST_DATA_DIR}/mnist_train.csv)

    find_program(CURL_EXECUTABLE curl)

    if (NOT CURL_EXECUTABLE)
        message(WARNING
            "BUILD_MNIST=ON but 'curl' not found. "
            "Please install curl or manually put MNIST CSVs in: ${MNIST_DATA_DIR}")
    else()
        add_custom_command(
            OUTPUT "${MNIST_TRAIN_CSV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MNIST_DATA_DIR}"
            COMMAND ${CURL_EXECUTABLE}
                    -L
                    -o "${MNIST_ZIP}"
                    https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv
            COMMAND ${CMAKE_COMMAND} -E tar xvf "${MNIST_ZIP}"
            WORKING_DIRECTORY "${MNIST_DATA_DIR}"
            COMMAND ${CMAKE_COMMAND} -E rm -f "${MNIST_ZIP}"
            COMMENT "Downloading MNIST CSV to ${MNIST_DATA_DIR}"
        )

        add_custom_target(mnist_data
            DEPENDS "${MNIST_TRAIN_CSV}"
        )

        # Ensure mnistClassifier can only build after mnist_data
        add_dependencies(mnistClassifier mnist_data)
    endif()
endif()
