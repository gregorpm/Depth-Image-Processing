include(../copy_file.cmake)
include(../copy_folder.cmake)

# Copy and Install HDF5 DLLs
copy_file("hdf5.dll" "${HDF5_INCLUDE_DIR}/../bin")
copy_file("hdf5_cpp.dll" "${HDF5_INCLUDE_DIR}/../bin")
copy_file("szip.dll" "${HDF5_INCLUDE_DIR}/../bin")
copy_file("zlib.dll" "${HDF5_INCLUDE_DIR}/../bin")

# Copy and Install GLEW DLLs
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  copy_file("glew32.dll" "${GLEW_INCLUDE_DIRS}/../bin/Release/x64")
else()
  copy_file("glew32.dll" "${GLEW_INCLUDE_DIRS}/../bin/Release/Win32")
endif()

# Copy and Install OpenNI2 DLLs
copy_file("OpenNI2.dll" "${OPENNI2_INCLUDE_DIRS}/../Redist")
copy_folder("OpenNI2" "${OPENNI2_INCLUDE_DIRS}/../Redist")

# Copy and Install OpenCV DLLs
set(OpenCV_NUMBER
    "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  copy_file("opencv_core${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_core${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_highgui${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_highgui${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_imgproc${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_imgproc${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_objdetect${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x64/vc11/bin")
  copy_file("opencv_objdetect${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x64/vc11/bin")
else()
  copy_file("opencv_core${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_core${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_highgui${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_highgui${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_imgproc${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_imgproc${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_objdetect${OpenCV_NUMBER}.dll" "${OpenCV_DIR}/x86/vc11/bin")
  copy_file("opencv_objdetect${OpenCV_NUMBER}d.dll" "${OpenCV_DIR}/x86/vc11/bin")
endif()
