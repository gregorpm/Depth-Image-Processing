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

#ifndef DIP_IO_HDF5DUMPER_H
#define DIP_IO_HDF5DUMPER_H

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

#include <dip/common/macros.h>
#include <dip/io/hdf5wrapper.h>

namespace dip {

typedef struct {
  char *name, *group;
  void *buffer;
  hsize_t *dimensions;
  int number_dimensions;
  hid_t datatype;
} HDF5Data;

class HDF5Dumper {
public:
  HDF5Dumper(HDF5Wrapper *hdf5);
  ~HDF5Dumper();

  void Write(const char *name, const char *group, const void *buffer,
             int bytes, const hsize_t *dimensions, int number_dimensions,
             hid_t datatype);

  friend void dump_data(HDF5Dumper *dumper);

private:
  HDF5Wrapper *hdf5_;
  std::queue<HDF5Data> data_;

  std::thread worker_;
  std::mutex mtx_;
  std::condition_variable cv_;

  bool running_;

  DISALLOW_COPY_AND_ASSIGN(HDF5Dumper);
};

} // namespace dip

#endif // DIP_IO_HDF5DUMPER_H
