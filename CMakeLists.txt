cmake_minimum_required(VERSION 3.18)
project(KNN_CUDA_Project LANGUAGES CXX CUDA)

# Set the C++ and CUDA standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Include directories
include_directories(include)

# Add the executable, linking the source files
add_executable(knn_cuda_project src/main.cpp src/knn_classifier.cpp src/csv_utils.cpp src/knn_cuda.cu)

# Specify target properties for CUDA
set_target_properties(knn_cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
