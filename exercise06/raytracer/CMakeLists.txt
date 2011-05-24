cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIR})

find_package(CUDA)

find_package(GLog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(YAMLCPP REQUIRED)
include_directories(${YAMLCPP_INCLUDE_DIR})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})

include_directories("src")



if(CUDA_FOUND)
    
	set (CUDA_NVCC_FLAGS "-arch=sm_20")
	include_directories(${CUDA_INCLUDE_DIRS}) 
	
	cuda_add_executable(rtGPU src/main.cc src/parser.cc src/raytracer.cu src/ppm_writer.cc src/types.cc src/windowHandling.cc) 
	target_link_libraries(rtGPU ${GLOG_LIBRARY})
	target_link_libraries(rtGPU ${GFLAGS_LIBRARY})
	target_link_libraries(rtGPU ${YAMLCPP_LIBRARY})
	target_link_libraries(rtGPU ${GLUT_LIBRARIES})

	configure_file(src/raytracer.cu src/raytracer.cc COPYONLY)
	add_executable(rtCPU src/main.cc src/parser.cc src/raytracer.cc src/ppm_writer.cc src/types.cc src/windowHandling.cc) 
	ADD_DEFINITIONS(-DNO_CUDA)
	set_target_properties (rtCPU PROPERTIES COMPILE_FLAGS "-DNO_CUDA")
	target_link_libraries(rtCPU ${GLOG_LIBRARY})
	target_link_libraries(rtCPU ${GFLAGS_LIBRARY})
	target_link_libraries(rtCPU ${YAMLCPP_LIBRARY})
	target_link_libraries(rtCPU ${GLUT_LIBRARIES})
	
	
endif()

if (NOT CUDA_FOUND)
	add_executable(rt src/main.cc src/parser.cc src/raytracer.cc src/ppm_writer.cc src/types.cc) 
	set_target_properties (rt PROPERTIES COMPILE_FLAGS "-DNO_CUDA")
	target_link_libraries(rt ${GLOG_LIBRARY})
	target_link_libraries(rt ${GFLAGS_LIBRARY})
	target_link_libraries(rt ${YAMLCPP_LIBRARY})
endif()



# common flags
SET(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas -Wno-long-long -Wno-variadic-macros")
SET(CMAKE_CXX_FLAGS_RELEASE "-DNEXPENSIVE -O2 -g0")
SET(CMAKE_CXX_FLAGS_DEBUG "-DWARNINGS -O0 -g3 -fno-inline")
