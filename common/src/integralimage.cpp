/*
Copyright (c) 2013, Gregory P. Meyer
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

#include <dip/common/integralimage.h>

namespace dip {

IntegralImage::~IntegralImage() {
  delete [] integral_;
}

void IntegralImage::Create(int width, int height, int *image, int mask) {
  if (size_ < (width * height)) {
    size_ = width * height;

    if (integral_ != NULL)
      delete [] integral_;

    integral_ = new int[size_];
  }

  int sum = 0;
  for (int x = 0; x < width; x++) {
    if (image[x] == mask)
      sum++;

    integral_[x] = sum;
  }

  for (int y = 1; y < height; y++) {
    sum = 0;
    for (int x = 0; x < width; x++) {
      int i = x + y * width;

      if (image[i] == mask)
        sum++;

      integral_[i] = integral_[i - width] + sum;
    }
  }
}

int IntegralImage::Sum(int x, int y, int width, int height, int half_window) {
  int left = MAX(x - half_window - 1, 0);
  int top = MAX(y - half_window - 1, 0);
  int right = MIN(x + half_window, width - 1);
  int bottom = MIN(y + half_window, height - 1);

  int i = left + top * width;
  int dx = right - left;
  int dy = (bottom - top) * width;

  int sum = integral_[i + dy + dx] -  // Bottom-Right Corner
            integral_[i + dy] -       // Bottom-Left Corner
            integral_[i + dx] +       // Top-Right Corner
            integral_[i];             // Top-Left Corner

  return sum;
}

} // namespace dip
