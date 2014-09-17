set(MODULE_NAME surface)

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
    include/dip/${MODULE_NAME}/marchingcubes.h
    include/dip/${MODULE_NAME}/mesh.h
    include/dip/${MODULE_NAME}/raycasting.h
    include/dip/${MODULE_NAME}/volumetric.h
    include/dip/${MODULE_NAME}/voxel.h
    )

set(SRCS
    src/marchingcubes.cpp
    src/raycasting.cpp
    src/raycasting.cu
    src/volumetric.cpp
    src/volumetric.cu
    )

set(LIBS
    ${MODULE_DEPENDENCIES}
    )

cuda_add_library(${MODULE_NAME} ${SRCS} ${INCS})
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
