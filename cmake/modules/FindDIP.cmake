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
  find_library(DIP_${MODULE}_LIBRARY
               NAMES ${MODULE}
               PATHS /usr/local/lib/
                     "$ENV{PROGRAMFILES}/dip/lib"
                     "$ENV{PROGRAMW6432}/dip/lib"
               )

  set(DIP_LIBRARIES
      ${DIP_LIBRARIES}
      ${DIP_${MODULE}_LIBRARY}
      )

  set(DIP_LIBRARY_VARS
      ${DIP_LIBRARY_VARS}
      DIP_${MODULE}_LIBRARY
      )
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DIP DEFAULT_MSG
                                  DIP_INCLUDE_DIR ${DIP_LIBRARY_VARS})
