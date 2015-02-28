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

#include <dip/io/hdf5wrapper.h>

#include <string>

using namespace std;

namespace dip {

HDF5Wrapper::HDF5Wrapper(const char* file_name, int mode)
    : enabled_(false), h5file_(-1), compression_(0) {
  // Disable HDF5 Error Messages
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  // Open/Create HDF5 File
  switch (mode) {
    case READ_HDF5:
      h5file_ = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
      break;
    case MODIFY_HDF5:
      h5file_ = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
      if (h5file_ == -1)
        h5file_ = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      break;
    case CREATE_HDF5:
      h5file_ = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      break;
  }

  if (h5file_ != -1) {
    mode_ = mode;
    enabled_ = true;
  }
}

HDF5Wrapper::~HDF5Wrapper() {
  if (enabled_)
    H5Fclose(h5file_);
}

int HDF5Wrapper::Read(const char *name, const char *group, void *buffer,
                      const hsize_t *dimensions, int number_dimensions,
                      hid_t datatype) const {
  if (enabled_) {
    int status = ReadData(h5file_, name, group, buffer, dimensions,
                          number_dimensions, datatype);

    return status;
  }

  return -1;
}

int HDF5Wrapper::Read(const char *name, const char *group, void *variable,
                      hid_t datatype) const {
  if (enabled_) {
    hsize_t dimensions = 1;

    int status = ReadData(h5file_, name, group, variable,
                          &dimensions, 1, datatype);

    return status;
  }

  return -1;
}

int HDF5Wrapper::Write(const char *name, const char *group, const void *buffer,
                       const hsize_t *dimensions, int number_dimensions,
                       hid_t datatype) {
  if (enabled_ && (mode_ & (MODIFY_HDF5 | CREATE_HDF5))) {
    int status = WriteData(h5file_, name, group, buffer, dimensions,
                           number_dimensions, datatype);

    return status;
  }

  return -1;
}

int HDF5Wrapper::Write(const char *name, const char *group,
                       const void *variable, hid_t datatype) {
  if (enabled_ && (mode_ & (MODIFY_HDF5 | CREATE_HDF5))) {
    hsize_t dimensions = 1;

    int status = WriteData(h5file_, name, group, variable,
                           &dimensions, 1, datatype);

    return status;
  }

  return -1;
}

int HDF5Wrapper::Compression(int level) {
  if (enabled_ && (mode_ & (MODIFY_HDF5 | CREATE_HDF5))) {
    if ((level >=0) && (level <= 9)) {
      compression_ = level;
      return 0;
    }
  }

  return -1;
}

int HDF5Wrapper::ReadData(hid_t h5_file, const char *name, const char *group,
                          void *buffer, const hsize_t *dimensions,
                          int number_dimensions, hid_t datatype) const {
  herr_t status;
  hid_t data_set, data_space, data_group;

  // Open Group
  data_group = H5Gopen2(h5_file, group, H5P_DEFAULT);
  if (data_group == -1)
    return -1;

  // Open Data Set
  data_set = H5Dopen2(data_group, name, H5P_DEFAULT);
  if (data_set == -1)
    return -1;

  // Open Data Space
  data_space = H5Dget_space(data_set);
  if (data_space == -1)
    return -1;

  // Check Number of Dimensions
  if (H5Sget_simple_extent_ndims(data_space) != number_dimensions)
    return -1;

  // Check Number of Points
  hssize_t number_points = 1;

  for (int i = 0; i < number_dimensions; i++)
    number_points *= dimensions[i];

  if (H5Sget_simple_extent_npoints(data_space) != number_points)
    return -1;

  // Check Data-type
  if (!H5Tequal(H5Dget_type(data_set), datatype))
    return -1;

  // Read Data Set
  status = H5Dread(data_set, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

  if (status == -1)
    return -1;

  // Close Data Space
  H5Sclose(data_space);
  // Close Data Set
  H5Dclose(data_set);
  // Close Group
  H5Gclose(data_group);

  return 0;
}

int HDF5Wrapper::WriteData(hid_t h5_file, const char *name, const char *group,
                           const void *buffer, const hsize_t *dimensions,
                           int number_dimensions, hid_t datatype) {
  herr_t status;
  hid_t data_set, data_space, data_group;

  // Create Group
  if (CreateGroup(h5_file, group) == -1)
    return -1;

  // Open Group
  data_group = H5Gopen2(h5_file, group, H5P_DEFAULT);
  if (data_group == -1)
    return -1;

  // Open Data Set
  data_set = H5Dopen2(data_group, name, H5P_DEFAULT);
  if (data_set == -1) {
    // Create Data Space
    data_space = H5Screate_simple(number_dimensions, dimensions, NULL);

    if (data_space != -1) {
      // Create Properties for Compression
      hid_t property = H5P_DEFAULT;

      if (compression_ > 0) {
        property = H5Pcreate(H5P_DATASET_CREATE);

        hsize_t *chunk_sizes = new hsize_t[number_dimensions];
        for (int n = 0; n < number_dimensions; n++)
          chunk_sizes[n] = MAX(dimensions[n] / 8, 1);
        H5Pset_chunk(property, number_dimensions, chunk_sizes);
        delete [] chunk_sizes;

        H5Pset_deflate(property, compression_);
      }

      // Create Data Set
      data_set = H5Dcreate2(data_group, name, datatype, data_space,
                            H5P_DEFAULT, property, H5P_DEFAULT);

      if (data_set == -1)
        return -1;

      // Close Data Space
      H5Sclose(data_space);
    }
    else {
      return -1;
    }
  }

  // Write Data Set
  status = H5Dwrite(data_set, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

  if (status == -1)
    return -1;

  // Close Date Set
  H5Dclose(data_set);
  // Close Group
  H5Gclose(data_group);

  return 0;
}

int HDF5Wrapper::CreateGroup(hid_t h5_file, const char *group) {
  // Parse Group name
  string group_name(group);

  int length = group_name.size();

  // Remove trailing forward slash
  if (group_name[length - 1] == '/')
    group_name.erase(length - 1);

  // Find last occurrence of a forward slash
  int slash_position = group_name.rfind("/");

  // Create Parent Group
  if ((slash_position != string::npos) && (slash_position != 0)) {
    string parent_name(group_name, 0, slash_position);

    if (CreateGroup(h5_file, parent_name.c_str()))
      return -1;
  }

  // Create Group
  hid_t data_group;

  data_group = H5Gopen2(h5_file, group_name.c_str(), H5P_DEFAULT);
  if (data_group == -1) {
    data_group = H5Gcreate2(h5_file, group_name.c_str(), H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);

    if (data_group == -1)
      return -1;
  }

  // Close Group
  H5Gclose(data_group);

  return 0;
}

} // namespace dip
