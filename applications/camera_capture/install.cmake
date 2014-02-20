include(../copy_dll.cmake)

# Copy and Install HDF5 DLLs
copy_dll("${HDF5_INCLUDE_DIR}/../bin/hdf5.dll")
copy_dll("${HDF5_INCLUDE_DIR}/../bin/hdf5_cpp.dll")
copy_dll("${HDF5_INCLUDE_DIR}/../bin/szip.dll")
copy_dll("${HDF5_INCLUDE_DIR}/../bin/zlib.dll")

# Copy and Install GLUT DLLs
copy_dll("${GLUT_INCLUDE_DIR}/../bin/freeglut.dll")

# Copy and Install OpenNI2 DLLs
copy_dll("${OPENNI2_INCLUDE_DIRS}/../Redist/OpenNI2.dll")

add_custom_command(TARGET ${APPICATION_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OPENNI2_INCLUDE_DIRS}/../Redist/OpenNI2"
                   "$<TARGET_FILE_DIR:${APPICATION_NAME}>/OpenNI2")
install(DIRECTORY "${OPENNI2_INCLUDE_DIRS}/../Redist/OpenNI2"
        DESTINATION "${BIN_INSTALL_DIR}")