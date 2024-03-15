cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

find_package(CUDAToolkit 11 REQUIRED)


project(PointCloudQuery LANGUAGES CXX CUDA)

## Use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)   

include(../../cmake/Glider-config.cmake)


set(BUILD_VBD OFF)
set(BUILD_PBD OFF)
set(BUILD_VBD_Cloth OFF)
find_package(Gaia)

include_directories(
	${Glider_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

file(GLOB SRC
    "*.h"
    "*.cpp"
	"*.c"
	"*.cu"
	"*.cuh"
	)

add_executable(PointCloudQuery 
	${SRC}
)


target_compile_options(PointCloudQuery PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
					   --default-stream per-thread
                       >)
#target_compile_definitions(P10_VBDDynamics PUBLIC USE_IPC_FRICTION USE_IPC_BOUNDARY_FRICTION)
#target_link_libraries(PointCloudQuery ${GAIA_LIBRARY})
