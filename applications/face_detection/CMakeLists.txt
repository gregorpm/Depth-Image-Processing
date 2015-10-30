SET(APPICATION_NAME face_detection)

SET(MODULE_DEPENDENCIES
    cameras
    common
    io
    segmentation
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
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${X11_LIBS}
    ${MODULE_DEPENDENCIES}
    )

add_executable(${APPICATION_NAME} ${SRCS} ${INCS})
target_link_libraries(${APPICATION_NAME} ${LIBS})

install(TARGETS ${APPICATION_NAME} DESTINATION ${BIN_INSTALL_DIR})

include(../copy_file.cmake)
copy_file("haarcascade_frontalface_default.xml" "${PROJECT_SOURCE_DIR}/data")

if(WIN32)
  include(install.cmake)
endif()
