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

#include <dip/filters/variance.h>

#include <math.h>

#include <dip/common/memory.h>
#include <dip/common/reduction.h>

namespace dip {

extern void VarianceKernel(int width, int height, const Depth *depth,
                           float *variance, float *std, float *valid);
extern void ThresholdKernel(float threshold, int width, int height,
                            const float *std, const Depth *depth,
                            Depth *filtered_depth);

Variance::Variance() : bytes_(0) {
  variance_ = NULL;
  std_ = NULL;
  valid_ = NULL;
}

Variance::~Variance() {
  if (variance_!= NULL)
    Deallocate((void*)variance_);
  if (std_!= NULL)
    Deallocate((void*)std_);
  if (valid_!= NULL)
    Deallocate((void*)valid_);
}

void Variance::Run(int width, int height, const Depth *depth,
                   Depth *filtered_depth) {
  int required_bytes = sizeof(float) * width * height;

  if (bytes_ < required_bytes) {
    bytes_ = required_bytes;

    if (variance_!= NULL)
      Deallocate((void*)variance_);
    if (std_!= NULL)
      Deallocate((void*)std_);
    if (valid_!= NULL)
      Deallocate((void*)valid_);

    Allocate((void**)&variance_, bytes_);
    Allocate((void**)&std_, bytes_);
    Allocate((void**)&valid_, bytes_);
  }

  VarianceKernel(width, height, depth, variance_, std_, valid_);

  float valid = Reduce(width * height, valid_);
  float mean_variance = Reduce(width * height, variance_) / valid;
  float mean_std = Reduce(width * height, std_) / valid;
  float std_std = sqrt(mean_variance - (mean_std * mean_std));

  ThresholdKernel(mean_std + 2.0f * std_std, width, height, std_,
                  depth, filtered_depth);
}

} // namespace dip
