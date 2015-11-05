function(copy_file filename directory)
  set(file_location ${directory}/${filename})

  # Copy file to working directory.
  add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD COMMAND
                      ${CMAKE_COMMAND} -E copy "${file_location}"
                      $<TARGET_FILE_DIR:${APPICATION_NAME}>)
  # Install file.
  install(FILES "${file_location}" DESTINATION ${BIN_INSTALL_DIR})
endfunction()
