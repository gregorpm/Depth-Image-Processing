set(MODULE_NAME projects)

SET(MODULE_DEPENDENCIES
    common
    filters
    point_cloud
    sampling
    segmentation
    surface
    registration
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
    include/
    )

include_directories(${INCLUDE_DIRS})

set(INCS
    include/dip/${MODULE_NAME}/facemodeling.h
    include/dip/${MODULE_NAME}/objectmodeling.h
    )

set(SRCS
    src/facemodeling.cpp
    src/objectmodeling.cpp
    )

set(LIBS
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
