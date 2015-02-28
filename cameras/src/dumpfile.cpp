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

#include <dip/cameras/dumpfile.h>

#include <stdio.h>

namespace dip {

DumpFile::DumpFile(const char *file_name) : enabled_(false), dump_file_(NULL) {
  // Initialize Variables
  for (int i = 0; i < SENSOR_TYPES; i++) {
    width_[i] = height_[i] = -1;
    fx_[i] = fy_[i] = -1.0f;
    count_[i] = 0;
  }

  if (file_name != NULL) {
    // Open dump file using HDF5 Wrapper.
    dump_file_ = new HDF5Wrapper(file_name, READ_HDF5);
    enabled_ = dump_file_->enabled();

    if (enabled_) {
      // Read image dimensions from file.
      dump_file_->Read("WIDTH", "/INFORMATION/DEPTH_SENSOR",
                       &width_[DEPTH_SENSOR], H5T_NATIVE_INT);
      dump_file_->Read("HEIGHT", "/INFORMATION/DEPTH_SENSOR",
                       &height_[DEPTH_SENSOR], H5T_NATIVE_INT);

      dump_file_->Read("WIDTH", "/INFORMATION/COLOR_SENSOR",
                       &width_[COLOR_SENSOR], H5T_NATIVE_INT);
      dump_file_->Read("HEIGHT", "/INFORMATION/COLOR_SENSOR",
                       &height_[COLOR_SENSOR], H5T_NATIVE_INT);

      // Read focal lengths from file.
      dump_file_->Read("FX", "/INFORMATION/DEPTH_SENSOR",
                       &fx_[DEPTH_SENSOR], H5T_NATIVE_FLOAT);
      dump_file_->Read("FY", "/INFORMATION/DEPTH_SENSOR",
                       &fy_[DEPTH_SENSOR], H5T_NATIVE_FLOAT);

      dump_file_->Read("FX", "/INFORMATION/COLOR_SENSOR",
                       &fx_[COLOR_SENSOR], H5T_NATIVE_FLOAT);
      dump_file_->Read("FY", "/INFORMATION/COLOR_SENSOR",
                       &fy_[COLOR_SENSOR], H5T_NATIVE_FLOAT);
    }
  }
}

DumpFile::~DumpFile() {
  if (dump_file_ != NULL)
    delete dump_file_;
}

int DumpFile::Update(Depth *depth) {
  if (enabled_) {
    char group[64];
    sprintf(group, "/FRAME%04d", count_[DEPTH_SENSOR]);

    hsize_t dimensions[2] = { (hsize_t)height_[DEPTH_SENSOR],
                              (hsize_t)width_[DEPTH_SENSOR] };

    // Read depth image from file.
    if (!dump_file_->Read("DEPTH", group, depth, dimensions, 2,
                          H5T_NATIVE_SHORT)) {
      // Update current depth frame.
      count_[DEPTH_SENSOR]++;
      return 0;
    }
  }

  return -1;
}

int DumpFile::Update(Color *color) {
  if (enabled_) {
    char group[64];
    sprintf(group, "/FRAME%04d", count_[COLOR_SENSOR]);

    hsize_t dimensions[3] = { (hsize_t)height_[COLOR_SENSOR],
                              (hsize_t)width_[COLOR_SENSOR], 3 };

    // Read color image from file.
    if (!dump_file_->Read("COLOR", group, color, dimensions, 3,
                          H5T_NATIVE_UCHAR)) {
      // Update current color frame.
      count_[COLOR_SENSOR]++;
      return 0;
    }
  }

  return -1;
}

} // namespace dip
