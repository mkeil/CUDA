cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIR})

find_package(CUDA REQUIRED)
set (CUDA_INCLUDE_DIRS "/usr/local/cuda")
set (CUDA_NVCC_FLAGS "-arch=sm_20")

find_package(GLog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(YAMLCPP REQUIRED)
include_directories(${YAMLCPP_INCLUDE_DIR})

include_directories("src")


cuda_add_executable(rtGPU src/main.cc src/parser.cc src/raytracer.cc src/ppm_writer.cc src/types.cc src/copyToDevice.cu) 
target_link_libraries(rtGPU ${GLOG_LIBRARY})
target_link_libraries(rtGPU ${GFLAGS_LIBRARY})
target_link_libraries(rtGPU ${YAMLCPP_LIBRARY})


cuda_add_executable(rt src/main.cc src/parser.cc src/raytracer.cc src/ppm_writer.cc src/types.cc OPTIONS "-DNO_CUDA") 
# ADD_DEFINITIONS(-DNO_CUDA)
set_target_properties (rt PROPERTIES COMPILE_FLAGS "-DNO_CUDA")
target_link_libraries(rt ${GLOG_LIBRARY})
target_link_libraries(rt ${GFLAGS_LIBRARY})
target_link_libraries(rt ${YAMLCPP_LIBRARY})

# common flags
SET(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas -Wno-long-long -Wno-variadic-macros")
SET(CMAKE_CXX_FLAGS_RELEASE "-DNEXPENSIVE -O2 -g0")
SET(CMAKE_CXX_FLAGS_DEBUG "-DWARNINGS -O0 -g3 -fno-inline")

