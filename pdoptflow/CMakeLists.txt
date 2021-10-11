#to use cuda 11: (on this machine currently /usr/bin/nvcc contains nvcc at version 10.1 not 11)
#cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.0 ..


#build instructions:
#cmake -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) ..
#make CXX=$(which g++) CC=$(which gcc)
cmake_minimum_required(VERSION 3.1)
project(1-wasserstein_dist)
INCLUDE(CheckCXXSourceCompiles)

find_package(CUDA REQUIRED)
#find_package(PythonInterp)# 3.6 REQUIRED)
#find_package(PythonLibs)# 3.6 REQUIRED)
add_subdirectory(pybind11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-G;-Xcompiler;-ggdb;-std=c++11)#manually specify
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-Xcompiler;-ggdb;-std=c++11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2)#;-std=c++11)#https://stackoverflow.com/questions/17703990/cuda-gdb-crashes-with-thrust-cuda-release-5-5
else ()
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-gencode;arch=compute_70,code=compute_70;-O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-std=c++11)
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_70,code=compute_70;-O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-std=c++11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-use_fast_math;-std=c++11)
endif ()

include(FindCUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
        "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
        )
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/)

FIND_PACKAGE(OpenMP)

set(CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
#include <omp.h>
int main() {
#if (_OPENMP >= 200805 || _MSC_VER >= 1500)
  return 0;
#else
  breaks_on_purpose
#endif
}
" OPENMP_VERSION)

if (OPENMP_VERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message("
OpenMP 3.0 not supported by the compiler (${CMAKE_CXX_COMPILER})!
use a C++ compiler with OpenMP 3.0 support (e.g., GCC >=4.4).
To use a different compiler, pass it to cmake in the variable CMAKE_CXX_COMPILER:
 cmake . -DCMAKE_CXX_COMPILER=g++-4.7
")
endif ()

#set(PROJECT_NAME 1-wasserstein_dist)
set(LEMON_DIRECTORY_NAME lemon-1.3.1-par)
#CXX
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -std=c++17 -pedantic -O3")
#set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -fopenmp -use_fast_math -O3 -Ofast")
set(CMAKE_CXX_FLAGS_DEBUG "-pedantic -g")

set(PROJECT_INCLUDE include/ ${CMAKE_CURRENT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build ${CMAKE_CURRENT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}  ${PYTHON_INCLUDE_DIRS} ./pybind11/include)

include_directories(${PROJECT_INCLUDE})

#include_directories(
#                ${PYTHON_INCLUDE_DIRS}
#                ./pybind11/include
#        )

##file(GLOB PROJECT_SRC_FILES#change this later to include files (want a header only library)
##  ${CMAKE_CURRENT_SOURCE_DIR}/include/loader/*.cu
##  #${CMAKE_CURRENT_SOURCE_DIR}/include/spanner/split_tree.cu)

#https://stackoverflow.com/questions/34799916/copy-file-from-source-directory-to-binary-directory-using-cmake
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/deps/${LEMON_DIRECTORY_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})



#https://stackoverflow.com/questions/3702115/creating-a-directory-in-cmake
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build)

##cuda_add_library(_${PROJECT_NAME} STATIC ${PROJECT_SRC_FILES})


cuda_add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/1-wasserstein_dist.cu)
####cuda_add_executable(k-experiments ${CMAKE_CURRENT_SOURCE_DIR}/k-experiments.cu)

cuda_add_library(W1 SHARED W1.cu)
pybind11_extension(W1)

target_link_libraries(W1
                ${PYTHON_LIBRARIES}
                #cudart
        -lemon -L${CMAKE_CURRENT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build/lemon)

set_target_properties(W1 PROPERTIES PREFIX "")

#https://stackoverflow.com/questions/15972898/cmake-how-to-run-a-add-custom-command-before-everything-else
#UNIX-specific:

add_custom_target(cmake_lemon COMMAND ${CMAKE_COMMAND} .. WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build)
add_custom_target(build_lemon COMMAND ${CMAKE_COMMAND} --build .  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build)

add_dependencies(build_lemon cmake_lemon)
#add_custom_target(build_lemon COMMAND ${CMAKE_COMMAND} .. && make WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build)

##add_dependencies(_${PROJECT_NAME} build_lemon)


add_dependencies(${PROJECT_NAME} build_lemon)
add_dependencies(W1 build_lemon)
####add_dependencies(k-experiments build_lemon)



#add_subdirectory(${PROJECT_SOURCE_DIR}/deps/lemon-1.3.1/)

##target_link_libraries(${PROJECT_NAME} _${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} -lemon -L${CMAKE_CURRENT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build/lemon)
####target_link_libraries(k-experiments -lemon -L${CMAKE_CURRENT_BINARY_DIR}/${LEMON_DIRECTORY_NAME}/build/lemon)

enable_testing()
add_subdirectory(testing_dir)

#include(ExternalProject)
#ExternalProject_Add(testing_dir SOURCE_DIR "testing_dir")
add_dependencies(879_test_battery build_lemon)
add_dependencies(more_tests build_lemon)
#################
