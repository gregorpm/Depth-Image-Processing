cmake_minimum_required(VERSION 2.8)
project(dip)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})

set(BIN_INSTALL_DIR "bin")
set(LIB_INSTALL_DIR "lib")
set(INCLUDE_INSTALL_DIR "include/${PROJECT_NAME}")

#add_definitions(-DSOFTKINETIC)

if(WIN32)
  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  set(BUILD_SHARED_LIBS true)
endif()

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_20")
if(UNIX)
  set(CUDA_PROPAGATE_HOST_FLAGS false)
endif()
# DepthSense
find_package(DepthSense REQUIRED)
# Eigen
find_package(Eigen REQUIRED)
# GLEW
find_package(GLEW REQUIRED)
# GLFW
find_package(GLFW REQUIRED)
# HDF5
find_package(HDF5 REQUIRED NO_MODULE)
# OpenCV
find_package(OpenCV REQUIRED)
# OpenGL
find_package(OpenGL REQUIRED)
# OpenNI
find_package(OpenNI2 REQUIRED)
# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
# X11
if(UNIX)
  find_package(X11 REQUIRED)
  set(X11_LIBS ${X11_Xrandr_LIB} ${X11_Xxf86vm_LIB} ${X11_Xi_LIB} ${X11_Xinerama_LIB} ${X11_Xcursor_LIB})
endif()

set(MODULE_NAMES
    cameras
    common
    filters
    io
    point_cloud
    projects
    registration
    sampling
    segmentation
    surface
    visualization
    )

foreach(MODULE_NAME ${MODULE_NAMES})
  add_subdirectory(${MODULE_NAME})
endforeach(MODULE_NAME)

add_subdirectory(applications)
