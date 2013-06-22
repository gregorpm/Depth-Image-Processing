set(APPLICATION_NAMES
    camera_viewer
    depth_capture
    face_modeling
    object_modeling
    )

foreach(APPLICATION_NAME ${APPLICATION_NAMES})
  add_subdirectory(${APPLICATION_NAME})
endforeach(APPLICATION_NAME)
