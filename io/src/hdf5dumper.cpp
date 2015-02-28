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

#include <dip/io/hdf5dumper.h>

#include <string>

using namespace std;

namespace dip {

const int kMaxLength = 256;

void dump_data(HDF5Dumper *dumper) {
  unique_lock<mutex> lck(dumper->mtx_);
  while(dumper->running_) {
    while(dumper->data_.empty() && dumper->running_) dumper->cv_.wait(lck);

    while(!dumper->data_.empty()) {
      HDF5Data data = dumper->data_.front();
      dumper->data_.pop();

      lck.unlock();
      dumper->hdf5_->Write(data.name, data.group, data.buffer, data.dimensions,
                           data.number_dimensions, data.datatype);

      delete [] data.name;
      delete [] data.group;
      delete [] data.buffer;
      delete [] data.dimensions;
      lck.lock();
    }
  }
}

HDF5Dumper::HDF5Dumper(HDF5Wrapper *hdf5) : hdf5_(hdf5), running_(true) {
  worker_ = thread(dump_data, this);
}

HDF5Dumper::~HDF5Dumper() {
  unique_lock<mutex> lck(mtx_);
  running_ = false;
  cv_.notify_one();

  lck.unlock();
  worker_.join();
}

void HDF5Dumper::Write(const char *name, const char *group, const void *buffer,
                       int bytes, const hsize_t *dimensions,
                       int number_dimensions, hid_t datatype) {
  HDF5Data data;

  data.name = new char[strnlen(name, kMaxLength) + 1];
  memcpy(data.name, name, sizeof(char) * strnlen(name, kMaxLength));
  data.name[strnlen(name, kMaxLength)] = '\0';

  data.group = new char[strnlen(group, kMaxLength) + 1];
  memcpy(data.group, group, sizeof(char) * strnlen(group, kMaxLength));
  data.group[strnlen(group, kMaxLength)] = '\0';

  data.buffer = new unsigned char[bytes];
  memcpy(data.buffer, buffer, bytes);

  data.dimensions = new hsize_t[number_dimensions];
  memcpy(data.dimensions, dimensions, sizeof(hsize_t) * number_dimensions);

  data.number_dimensions = number_dimensions;
  data.datatype = datatype;

  unique_lock<mutex> lck(mtx_);
  data_.push(data);
  cv_.notify_one();
}

} // namespace dip
