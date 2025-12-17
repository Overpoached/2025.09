cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Collect all .cpp files in the src directory
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Collect all .cu files (CUDA source files) in the src directory
file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

# Collect all .h files in the src directory
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add executable with all collected sources (C++, CUDA, and headers)
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES} ${HEADERS})

# Include CUDA headers for IDE IntelliSense
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Set CUDA architectures (adjust based on your GPU)
# Use only architectures supported by your CUDA toolkit version
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52;61;75;86"  # 52=GTX 9xx, 61=GTX 10xx, 75=RTX 20xx, 86=RTX 30xx
)

# Print collected files for debugging (optional)
message(STATUS "Found C++ source files: ${SOURCES}")
message(STATUS "Found CUDA source files: ${CUDA_SOURCES}")
message(STATUS "Found header files: ${HEADERS}")
