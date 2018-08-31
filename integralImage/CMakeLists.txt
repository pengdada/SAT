cmake_minimum_required(VERSION 2.8)
project(sat)

#set(CMAKE_CXX_FLAGS " -Wall ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native  ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native ${CMAKE_CXX_FLAGS_MINSIZEREL}")


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
	message(STATUS "###################find OPENMP ${OpenMP_CXX_FLAGS}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0/)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	set(CUDA_HOST_COMPILER g++)
	set(CUDA_VERBOSE_BUILD ON)	
endif(CUDA_FOUND)

if (UNIX AND NOT APPLE)
	list(APPEND CUDA_NVCC_FLAGS -ftz=true -prec-div=false -prec-sqrt=false  -Wno-deprecated-declarations "-w" -std=c++11)
	list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v  -Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant")
	list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-D_DEBUG;-O0;-g;-G;)
endif()


file(GLOB FILES_CPP "*.cpp")

message(STATUS "#################${FILES_CPP}")


file(GLOB CUDA_FILES "*.cu")
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	message(STATUS "################### CMAKE_BUILD_TYPE == Debug:   "OPTIONS -G"")
	CUDA_COMPILE(CU_O ${CUDA_FILES} "OPTIONS -G OPTIONS -g")
else()
	message(STATUS "################### CMAKE_BUILD_TYPE == Release:  ")
	CUDA_COMPILE(CU_O ${CUDA_FILES})
endif()

cuda_add_executable(sat ${FILES_CPP} ${CU_O})

target_link_libraries(sat ${CUDA_LIBRARY} ${OpenCV_LIBS} ${CUDA_cufft_LIBRARY})




