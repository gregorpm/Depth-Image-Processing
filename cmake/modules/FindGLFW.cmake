###############################################################################
# Find GLFW
#
# GLFW_FOUND - True if GLFW was found.
# GLFW_INCLUDE_DIRS - Directories containing the GLFW include files.
# GLFW_LIBRARIES - Libraries needed to use GLFW.

find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h
          HINTS /usr/local/include
          PATHS "$ENV{PROGRAMFILES}/glfw/include/"
                "$ENV{PROGRAMW6432}/glfw/include/")

find_library(GLFW_LIBRARY
             NAMES glfw3
             HINTS /usr/local/lib
             PATHS "$ENV{PROGRAMFILES}/glfw/lib-msvc110"
                   "$ENV{PROGRAMFILES}/glfw/lib-vc2012"
                   "$ENV{PROGRAMW6432}/glfw/lib-msvc110"
                   "$ENV{PROGRAMW6432}/glfw/lib-vc2012")

set(GLFW_INCLUDE_DIRS
    ${GLFW_INCLUDE_DIR}
    )

set(GLFW_LIBRARIES
    ${GLFW_LIBRARY}
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW DEFAULT_MSG
                                  GLFW_INCLUDE_DIR GLFW_LIBRARY)
