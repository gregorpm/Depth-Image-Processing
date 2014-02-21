function(copy_file path)
  # Copy file to working directory.
  add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD COMMAND
                      ${CMAKE_COMMAND} -E copy "${path}"
                      $<TARGET_FILE_DIR:${APPICATION_NAME}>)
  # Install file.
  install(FILES "${path}" DESTINATION ${BIN_INSTALL_DIR})
endfunction()
