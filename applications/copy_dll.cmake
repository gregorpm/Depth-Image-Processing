function(copy_dll path)
  # Copy DLL to working directory.
  add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD COMMAND
                      ${CMAKE_COMMAND} -E copy "${path}"
                      $<TARGET_FILE_DIR:${APPICATION_NAME}>)
  # Install DLL.
  install(FILES "${path}" DESTINATION ${BIN_INSTALL_DIR})
endfunction()
