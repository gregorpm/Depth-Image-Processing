function(copy_folder folder_name directory)
  add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_directory
                     "${directory}/${folder_name}"
                     "$<TARGET_FILE_DIR:${APPICATION_NAME}>/${folder_name}")

  install(DIRECTORY "${directory}/${folder_name}"
          DESTINATION "${BIN_INSTALL_DIR}")
endfunction()
