###############################################################################
# Find DepthSense SDK
#
# DEPTHSENSE_FOUND - True if DepthSense was found.
# DEPTHSENSE_INCLUDE_DIRS - Directories containing the DepthSense include 
#                           files.
# DEPTHSENSE_LIBRARIES - Libraries need to use DepthSense.

find_path(DEPTHSENSE_INCLUDE_DIR DepthSense.hxx
          HINTS /opt/softkinetic/DepthSenseSDK/include
          PATHS "$ENV{PROGRAMFILES}/SoftKinetic/DepthSenseSDK/include"
                "$ENV{PROGRAMW6432}/SoftKinetic/DepthSenseSDK/include")

find_library(DEPTHSENSE_LIBRARY
             NAMES DepthSense
             HINTS /opt/softkinetic/DepthSenseSDK/lib
             PATHS "$ENV{PROGRAMFILES}/SoftKinetic/DepthSenseSDK/lib"
                   "$ENV{PROGRAMW6432}/SoftKinetic/DepthSenseSDK/lib")

set(DEPTHSENSE_INCLUDE_DIRS
    ${DEPTHSENSE_INCLUDE_DIR}
    )

set(DEPTHSENSE_LIBRARIES
    ${DEPTHSENSE_LIBRARY}
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DEPTHSENSE DEFAULT_MSG
                                  DEPTHSENSE_INCLUDE_DIR DEPTHSENSE_LIBRARY)
