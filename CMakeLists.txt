cmake_minimum_required(VERSION 2.8)
project(dip)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

set(BIN_INSTALL_DIR "bin")
set(LIB_INSTALL_DIR "lib")
set(INCLUDE_INSTALL_DIR "include/${PROJECT_NAME}")

if(UNIX)
  set(BUILD_SHARED_LIBS true)
endif()

# CUDA
find_package(CUDA REQUIRED)
# Eigen
find_package(Eigen REQUIRED)
# GLUT
find_package(GLUT REQUIRED)
# HDF5
find_package(HDF5 REQUIRED)
# OpenGL
find_package(OpenGL REQUIRED)
# OpenNI
find_package(OpenNI2 REQUIRED)

set(MODULE_NAMES
    cameras
    common
    filters
    io
    point_cloud
    projects
    registration
    sampling
    surface
    visualization
    )

foreach(MODULE_NAME ${MODULE_NAMES})
  add_subdirectory(${MODULE_NAME})
endforeach(MODULE_NAME)

add_subdirectory(applications)
