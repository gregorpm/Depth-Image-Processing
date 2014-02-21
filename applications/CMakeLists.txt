set(APPLICATION_NAMES
    camera_capture
    camera_viewer
    face_detection
    face_modeling
    object_modeling
    )

foreach(APPLICATION_NAME ${APPLICATION_NAMES})
  add_subdirectory(${APPLICATION_NAME})
endforeach(APPLICATION_NAME)
