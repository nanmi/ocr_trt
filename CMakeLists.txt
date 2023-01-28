cmake_minimum_required(VERSION 3.10)

project(ocr_det)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

include_directories(${PROJECT_SOURCE_DIR})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda/cudnn
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# tensorrt
set(TRT_OSS_ROOT /home/zhangbo/libraries/TensorRT-8.4.1.5)
include_directories(${TRT_OSS_ROOT}/include)
link_directories(${TRT_OSS_ROOT}/lib)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# det
add_executable(ocr_det 
                        ${PROJECT_SOURCE_DIR}/ocr_det.cpp
                        ${PROJECT_SOURCE_DIR}/clipper.cpp
                        ${PROJECT_SOURCE_DIR}/postprocess_op.cpp
                        ${PROJECT_SOURCE_DIR}/utility.cpp
                        )
target_link_libraries(ocr_det nvinfer)
target_link_libraries(ocr_det cudart)
target_link_libraries(ocr_det ${OpenCV_LIBS} nvparsers cublas nvinfer_plugin cudnn)

# rec
add_executable(ocr_rec 
                        ${PROJECT_SOURCE_DIR}/ocr_rec.cpp
                        ${PROJECT_SOURCE_DIR}/clipper.cpp
                        ${PROJECT_SOURCE_DIR}/postprocess_op.cpp
                        ${PROJECT_SOURCE_DIR}/utility.cpp
                        )
target_link_libraries(ocr_rec nvinfer)
target_link_libraries(ocr_rec cudart)
target_link_libraries(ocr_rec ${OpenCV_LIBS} nvparsers cublas nvinfer_plugin cudnn)

add_definitions(-O2 -pthread)

