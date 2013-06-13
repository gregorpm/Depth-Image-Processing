SET(APPICATION_NAME camera_viewer)

SET(MODULE_DEPENDENCIES
    cameras
    common
    io
    visualization
    )

include(includes.cmake)

foreach(DEPENDENT ${MODULE_DEPENDENCIES})
  include(${PROJECT_SOURCE_DIR}/${DEPENDENT}/includes.cmake)

  set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/${DEPENDENT}/include
    )
endforeach(DEPENDENT)

set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    )

include_directories(${INCLUDE_DIRS})

set(INCS
    )

set(SRCS
    ${APPICATION_NAME}.cpp
    )

set(LIBS
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${MODULE_DEPENDENCIES}
    )

add_executable(${APPICATION_NAME} ${SRCS} ${INCS})
target_link_libraries(${APPICATION_NAME} ${LIBS})

install(TARGETS ${APPICATION_NAME} DESTINATION ${BIN_INSTALL_DIR})

if(WIN32)
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
    add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${PROJECT_SOURCE_DIR}/3rdparty/GLUT/bin/x86"
                       $<TARGET_FILE_DIR:${APPICATION_NAME}>)
    add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x86"
                       $<TARGET_FILE_DIR:${APPICATION_NAME}>)
    add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/bin/x86"
                       $<TARGET_FILE_DIR:${APPICATION_NAME}>)

    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/GLUT/bin/x86/freeglut.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x86/hdf5.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x86/hdf5_cpp.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x86/szip.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x86/zlib.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/bin/x86/OpenNI2.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(DIRECTORY  "${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/bin/x86/OpenNI2"
            DESTINATION ${BIN_INSTALL_DIR})
  else()
    add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${PROJECT_SOURCE_DIR}/3rdparty/GLUT/bin/x64"
                       $<TARGET_FILE_DIR:${APPICATION_NAME}>)
    add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x64"
                       $<TARGET_FILE_DIR:${APPICATION_NAME}>)
    add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/bin/x64"
                       $<TARGET_FILE_DIR:${APPICATION_NAME}>)

    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/GLUT/bin/x64/freeglut.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x64/hdf5.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x64/hdf5_cpp.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x64/szip.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/HDF5/bin/x64/zlib.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/bin/x64/OpenNI2.dll"
            DESTINATION ${BIN_INSTALL_DIR})
    install(DIRECTORY  "${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/bin/x64/OpenNI2"
            DESTINATION ${BIN_INSTALL_DIR})
  endif()
endif()
