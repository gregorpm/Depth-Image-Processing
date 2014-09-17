set(MODULE_NAME segmentation)

SET(MODULE_DEPENDENCIES
    common
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
    include/
    )

include_directories(${INCLUDE_DIRS})

set(INCS
    include/dip/${MODULE_NAME}/connectedcomponents.h
    include/dip/${MODULE_NAME}/facemasker.h
    include/dip/${MODULE_NAME}/headsegmenter.h
    )

set(SRCS
    src/connectedcomponents.cpp
    src/facemasker.cpp
    src/headsegmenter.cpp
    )

set(LIBS
    ${OpenCV_LIBRARIES}
    ${MODULE_DEPENDENCIES}
    )

add_library(${MODULE_NAME} ${SRCS} ${INCS})
target_link_libraries(${MODULE_NAME} ${LIBS})

if(WIN32)
    install(TARGETS ${MODULE_NAME} DESTINATION ${LIB_INSTALL_DIR}/Release
            CONFIGURATIONS Release)
    install(TARGETS ${MODULE_NAME} DESTINATION ${LIB_INSTALL_DIR}/Debug
            CONFIGURATIONS Debug)
endif()

if(UNIX)
    install(TARGETS ${MODULE_NAME} DESTINATION ${LIB_INSTALL_DIR})
endif()

install(FILES ${INCS} DESTINATION ${INCLUDE_INSTALL_DIR}/${MODULE_NAME})
