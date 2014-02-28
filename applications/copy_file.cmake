function(copy_file filename directory)
  # Clear file location.
  set(file_location "file_location-NOTFOUND")

  # Find file.
  find_file(file_location
    NAMES ${filename}
    PATHS ${directory}
    PATH_SUFFIXES "Release" "Debug"
  )

  # Copy file to working directory.
  add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD COMMAND
                      ${CMAKE_COMMAND} -E copy "${file_location}"
                      $<TARGET_FILE_DIR:${APPICATION_NAME}>)
  # Install file.
  install(FILES "${file_location}" DESTINATION ${BIN_INSTALL_DIR})
endfunction()
