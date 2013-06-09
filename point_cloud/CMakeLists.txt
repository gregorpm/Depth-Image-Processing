set(MODULE_NAME point_cloud)

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
    include/dip/${MODULE_NAME}/backprojection.h
    include/dip/${MODULE_NAME}/centroid.h
    )

set(SRCS
    src/backprojection.cpp
    src/backprojection.cu
    src/centroid.cpp
    src/centroid.cu
    )

set(LIBS
    ${MODULE_DEPENDENCIES}
    )

cuda_add_library(${MODULE_NAME} ${SRCS} ${INCS})
target_link_libraries(${MODULE_NAME} ${LIBS})

install(TARGETS ${MODULE_NAME} DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${INCS} DESTINATION ${INCLUDE_INSTALL_DIR}/${MODULE_NAME})
