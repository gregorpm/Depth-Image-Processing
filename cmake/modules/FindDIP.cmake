###############################################################################
# Find Depth-Image-Processing (DIP)
#
# DIP_FOUND - True if DIP was found.
# DIP_INCLUDE_DIR - Directory containing the DIP include files.
# DIP_LIBRARIES - Libraries need to use DIP.

find_path(DIP_INCLUDE_DIR dip
          HINTS /usr/local/include/
                "$ENV{PROGRAMFILES}/dip/include"
                "$ENV{PROGRAMW6432}/dip/include"
          )

set(DIP_MODULES
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

foreach(MODULE ${DIP_MODULES})
  find_library(DIP_${MODULE}_LIBRARY_DEBUG
               NAMES ${MODULE}
               PATHS /usr/local/lib
                     "$ENV{PROGRAMFILES}/dip/lib/Debug"
                     "$ENV{PROGRAMW6432}/dip/lib/Debug"
               )

  find_library(DIP_${MODULE}_LIBRARY_RELEASE
               NAMES ${MODULE}
               PATHS /usr/local/lib
                     "$ENV{PROGRAMFILES}/dip/lib/Release"
                     "$ENV{PROGRAMW6432}/dip/lib/Release"
               )

  set(DIP_LIBRARIES
      ${DIP_LIBRARIES}
      debug ${DIP_${MODULE}_LIBRARY_DEBUG}
      optimized ${DIP_${MODULE}_LIBRARY_RELEASE}
      )

  set(DIP_LIBRARY_VARS
      ${DIP_LIBRARY_VARS}
      DIP_${MODULE}_LIBRARY_DEBUG
      DIP_${MODULE}_LIBRARY_RELEASE
      )
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DIP DEFAULT_MSG
                                  DIP_INCLUDE_DIR ${DIP_LIBRARY_VARS})
