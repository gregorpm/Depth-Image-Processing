/*
Copyright (c) 2013-2015, Gregory P. Meyer
                         University of Illinois Board of Trustees
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Wrapper around HDF5 C API to simplify reading, modifying, and creating
// HDF5 files. Simplifies the reading and writing of arrays and variables.

#ifndef DIP_IO_HDF5WRAPPER_H
#define DIP_IO_HDF5WRAPPER_H

#include <hdf5.h>

#include <dip/common/macros.h>

namespace dip {

// HDF5 file access modes.
enum HDF5_MODES {
  READ_HDF5 = 1,    // Read an existing HDF5 file.
  MODIFY_HDF5 = 2,  // Read/Write an existing HDF5 file.
  CREATE_HDF5 = 4,  // Creates a new HDF5 file.
};

class HDF5Wrapper {
public:
  // Opens a HDF5 file.
  //  file_name - Name of HDF5 file.
  //  mode      - File access mode (READ_HDF5, MODIFY_HDF5, CREATE_HDF5).
  HDF5Wrapper(const char* file_name, int mode);
  ~HDF5Wrapper();

  // Reads an array from the HDF5 file.
  //  name              - Name of array to be read.
  //  group             - Location of array within the HDF5 file.
  //  buffer            - Buffer to hold the array.
  //  dimensions        - A 1D array containing the size of each dimension.
  //  number_dimensions - The number of dimensions in the array.
  //                      (dimensions should have this many elements)
  //  datatype          - Datatype of elements stored in the array.
  //                      Must use HDF5 predefined datatypes:
  //                        float = H5T_NATIVE_FLOAT
  //                        int   = H5T_NATIVE_INT
  //                        etc.
  // Returns zero when successful.
  int Read(const char *name, const char *group, void *buffer,
           const hsize_t *dimensions, int number_dimensions,
           hid_t datatype) const;

  // Reads a variable from the HDF5 file.
  //  name              - Name of variable to be read.
  //  group             - Location of variable within the HDF5 file.
  //  variable          - Buffer to hold the variable.
  //  datatype          - Datatype of variable.
  //                      Must use HDF5 predefined datatypes:
  //                        float = H5T_NATIVE_FLOAT
  //                        int   = H5T_NATIVE_INT
  //                        etc.
  // Returns zero when successful.
  int Read(const char *name, const char *group, void *variable,
           hid_t datatype) const;

  // Writes an array into the HDF5 file.
  //  name              - Name of array to be written.
  //  group             - Location of array within the HDF5 file.
  //  buffer            - Buffer that holds the array.
  //  dimensions        - A 1D array containing the size of each dimension.
  //  number_dimensions - The number of dimensions in the array.
  //                      (dimensions should have this many elements)
  //  datatype          - Datatype of elements stored in the array.
  //                      Must use HDF5 predefined datatypes:
  //                        float = H5T_NATIVE_FLOAT
  //                        int   = H5T_NATIVE_INT
  //                        etc.
  // Returns zero when successful.
  int Write(const char *name, const char *group, const void *buffer,
            const hsize_t *dimensions, int number_dimensions,
            hid_t datatype);

  // Writes a variable into the HDF5 file.
  //  name              - Name of variable to be written.
  //  group             - Location of variable within the HDF5 file.
  //  variable          - Buffer that holds the variable.
  //  datatype          - Datatype of variable.
  //                      Must use HDF5 predefined datatypes:
  //                        float = H5T_NATIVE_FLOAT
  //                        int   = H5T_NATIVE_INT
  //                        etc.
  // Returns zero when successful.
  int Write(const char *name, const char *group, const void *variable,
            hid_t datatype);

  // Sets compression amount.
  //  level - Compression level (0 = No Compress to 9 = Best Compression).
  // Returns zero when successful.
  int Compression(int level);

  // Returns true if the HDF5 file was successfully opened.
  bool enabled() const { return enabled_; }

  // Return HDF5 file id.
  hid_t h5file() const { return h5file_; }

private:
  // Reads data from the HDF5 file.
  //  h5_file           - HDF5 file id.
  //  name              - Name of data to be read.
  //  group             - Location of data within the HDF5 file.
  //  buffer            - Buffer to hold the data.
  //  dimensions        - A 1D array containing the size of each dimension.
  //  number_dimensions - The number of dimensions in the array.
  //                      (dimensions should have this many elements)
  //  datatype          - Datatype of elements stored in the array.
  //                      Must use HDF5 predefined datatypes:
  //                        float = H5T_NATIVE_FLOAT
  //                        int   = H5T_NATIVE_INT
  //                        etc.
  // Returns zero when successful.
  int ReadData(hid_t h5_file, const char *name, const char *group,
               void *buffer, const hsize_t *dimensions, int number_dimensions,
               hid_t datatype) const;

  // Writes data into the HDF5 file.
  //  h5_file           - HDF5 file id.
  //  name              - Name of data to be written.
  //  group             - Location of data within the HDF5 file.
  //  buffer            - Buffer that holds the data.
  //  dimensions        - A 1D array containing the size of each dimension.
  //  number_dimensions - The number of dimensions in the array.
  //                      (dimensions should have this many elements)
  //  datatype          - Datatype of elements stored in the array.
  //                      Must use HDF5 predefined datatypes:
  //                        float = H5T_NATIVE_FLOAT
  //                        int   = H5T_NATIVE_INT
  //                        etc.
  // Returns zero when successful.
  int WriteData(hid_t h5_file, const char *name, const char *group,
                const void *buffer, const hsize_t *dimensions,
                int number_dimensions, hid_t datatype);

  // Creates group in the HDF5 file.
  //  h5_file - HDF5 file id.
  //  group   - name of group to be created.
  // Returns zero when successful.
  int CreateGroup(hid_t h5_file, const char *group);

  int mode_;
  bool enabled_;

  int compression_;

  // Id of HDF5 file.
  hid_t h5file_;

  DISALLOW_COPY_AND_ASSIGN(HDF5Wrapper);
};

} // namespace dip

#endif // DIP_IO_HDF5WRAPPER_H
