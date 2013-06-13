###############################################################################
# Find OpenNI2
#
# OPENNI2_FOUND - True if OpenNI2 was found.
# OPENNI2_INCLUDE_DIRS - Directories containing the OpenNI2 include files.
# OPENNI2_LIBRARIES - Libraries need to use OpenNI2.

find_path(OPENNI2_INCLUDE_DIR OpenNI.h
          HINTS ${PC_OPENNI2_INCLUDEDIR} ${PC_OPENNI2_INCLUDE_DIRS}
                /usr/include/openni2 /usr/include/ni2
          PATHS "$ENV{PROGRAMFILES}/OpenNI2/include"
                "$ENV{PROGRAMW6432}/OpenNI2/include"
          PATH_SUFFIXES openni2 ni2)

find_library(OPENNI2_LIBRARY
             NAMES OpenNI2
             HINTS ${PC_OPENNI2_LIBDIR} ${PC_OPENNI2_LIBRARY_DIRS}
                   /usr/lib
             PATHS "$ENV{PROGRAMFILES}/OpenNI2/Lib${OPENNI2_SUFFIX}"
                   "$ENV{PROGRAMW6432}/OpenNI2/Lib${OPENNI2_SUFFIX}"
                   "$ENV{PROGRAMW6432}/OpenNI2"
             PATH_SUFFIXES lib lib64)

set(OPENNI2_INCLUDE_DIRS
    ${OPENNI2_INCLUDE_DIR}
    )

set(OPENNI2_LIBRARIES
    ${OPENNI2_LIBRARY}
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENNI2 DEFAULT_MSG
                                  OPENNI2_INCLUDE_DIR OPENNI2_LIBRARY)
