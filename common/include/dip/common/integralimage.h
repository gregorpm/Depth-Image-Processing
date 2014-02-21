/*
Copyright (c) 2013-2014, Gregory P. Meyer
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

// This class creates an integral image from a binary image.

#ifndef DIP_COMMON_INTEGRAL_IMAGE_H
#define DIP_COMMON_INTEGRAL_IMAGE_H

#include <dip/common/macros.h>
#include <dip/common/types.h>

namespace dip {

class IntegralImage {
public:
  IntegralImage() : integral_(NULL), size_(0) {}
  ~IntegralImage();

  // Create Integral Image
  //  width & height - The dimensions of the image.
  //  image - Input image. 
  //  mask  - Foreground mask. Pixels that match the mask will be considered
  //          foreground, everything else is considered background.
  void Create(int width, int height, int *image, int mask);

  // Number of foreground pixels within a square window.
  //  x & y          - x and y position within the image.
  //  width & height - The dimensions of the image.
  //  half_window    - The half width of the window.
  int Sum(int x, int y, int width, int height, int half_window);

private:
  // Integral Image
  int *integral_;
  int size_;

  DISALLOW_COPY_AND_ASSIGN(IntegralImage);
};

} // namespace dip

#endif // DIP_COMMON_INTEGRAL_IMAGE_H
