cmake_minimum_required(VERSION 3.18)
project(poker-probability CUDA CXX)

# Set configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Source files
file(GLOB CPP_SOURCES
    "src/threads.cpp"
    "src/simulation.cpp"
)

# Explicitly set CUDA source instead of using GLOB
set(CUDA_SOURCES
    "src/cuda_simrun.cu"
)

file(GLOB HEADERS
    "src/*.h"
)

# Create static library for common code
add_library(poker-lib STATIC ${CPP_SOURCES} ${HEADERS})

# Include directories for the library
target_include_directories(poker-lib PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src
)

# Link libraries for the library
target_link_libraries(poker-lib PRIVATE
    CUDA::cudart
    Threads::Threads
)

# CUDA specific settings for the library
set_target_properties(poker-lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;86"
)

# Create static library for CUDA (modified to ensure source is added)
add_library(poker_cuda STATIC ${CUDA_SOURCES})
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Include directories for the CUDA static library
target_include_directories(poker_cuda PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src
)

# Link CUDA static library with CUDA runtime
target_link_libraries(poker_cuda PRIVATE CUDA::cudart)

# Link CUDA static library with the static library
target_link_libraries(poker-lib PRIVATE poker_cuda)

# Create main executable
add_executable(poker-probability 
    src/main.cpp
    src/simulation.cpp
)

# Link main executable with the library
target_link_libraries(poker-probability PRIVATE poker-lib poker_cuda)

# Include directories for the main executable
target_include_directories(poker-probability PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src
)

# Create test executable
add_executable(test_hand src/test_hand.cpp)

# Link test executable with the library
target_link_libraries(test_hand PRIVATE poker-lib poker_cuda)

# Include directories for the test executable
target_include_directories(test_hand PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src
)

# Add optimization flags for MSVC and CUDA compilers
if (MSVC)
    # Debug build: keep runtime checks, disable optimizations
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Zi /Od /RTC1")
    
    # Release build: full optimization, no runtime checks
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /Oi /Ot /DNDEBUG")
    
    # Disable specific warnings
    add_compile_options(/wd4820 /wd4710 /wd4711)
else()
    target_compile_options(poker-lib PRIVATE -O3 -funroll-loops)
    target_compile_options(poker-probability PRIVATE -O3 -funroll-loops)
    target_compile_options(test_hand PRIVATE -O3 -funroll-loops)
endif()

# CUDA specific optimization flags
if (MSVC)
    # Debug build
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DCUDA_DEBUG_ENABLED")
    
    # Release build
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -funroll-loops")
endif()

# Remove potentially conflicting flags
string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# CUDA specific optimization flags
set_target_properties(poker_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "60;70;75;86"
)

# Ensure CUDA separable compilation is enabled
set_target_properties(poker-probability PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_hand PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Ensure CUDA device symbols are resolved
set_target_properties(poker-probability PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_hand PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Add compilation flags for release and debug modes
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -DCUDA_DEBUG_ENABLED")