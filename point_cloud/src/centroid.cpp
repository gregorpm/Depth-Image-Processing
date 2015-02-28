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

#include <dip/point_cloud/centroid.h>

#include <dip/common/memory.h>
#include <dip/common/reduction.h>

namespace dip {

extern void CentroidKernel(int width, int height, Vertices vertices,
                           float *buffer[4]);

Centroid::Centroid() : bytes_(0) {
  for (int n = 0; n < 4; n++)
    buffer_[n] = NULL;
}

Centroid::~Centroid() {
  for (int n = 0; n < 4; n++) {
    if (buffer_[n] != NULL)
      Deallocate((void*)buffer_[n]);
  }
}

Vertex Centroid::Run(int width, int height, Vertices vertices) {
  int required_bytes = sizeof(float) * width * height;

  if (bytes_ < required_bytes) {
    bytes_ = required_bytes;

    for (int n = 0; n < 4; n++) {
      if (buffer_[n] != NULL)
        Deallocate((void*)buffer_[n]);

      Allocate((void**)&(buffer_[n]), bytes_);
    }
  }

  CentroidKernel(width, height, vertices, buffer_);

  // Compute Centroid
  Vertex center;
  int count;

  center.x = Reduce(width * height, buffer_[0]);
  center.y = Reduce(width * height, buffer_[1]);
  center.z = Reduce(width * height, buffer_[2]);
  count = Reduce(width * height, buffer_[3]);

  center.x /= count;
  center.y /= count;
  center.z /= count;

  return center;
}

} // namespace dip
