###############################################################################
# Find GLEW
#
# GLEW_FOUND - True if GLEW was found.
# GLEW_INCLUDE_DIRS - Directories containing the GLEW include files.
# GLEW_LIBRARIES - Libraries needed to use GLEW.

find_path(GLEW_INCLUDE_DIR GL/glew.h
          HINTS /usr/include
          PATHS "$ENV{PROGRAMFILES}/glew/include"
                "$ENV{PROGRAMW6432}/glew/include")

find_library(GLEW_LIBRARY
             NAMES GLEW glew32
             HINTS /usr/lib
                   /usr/lib64
             PATHS "$ENV{PROGRAMFILES}/glew/lib/Release/Win32"
                   "$ENV{PROGRAMW6432}/glew/lib/Release/x64")

set(GLEW_INCLUDE_DIRS
    ${GLEW_INCLUDE_DIR}
    )

set(GLEW_LIBRARIES
    ${GLEW_LIBRARY}
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW DEFAULT_MSG
                                  GLEW_INCLUDE_DIR GLEW_LIBRARY)
