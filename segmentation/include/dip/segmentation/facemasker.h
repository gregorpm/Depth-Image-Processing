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

#ifndef DIP_SEGMENTATION_FACEMASKER_H
#define DIP_SEGMENTATION_FACEMASKER_H

#include <dip/common/types.h>
#include <dip/common/macros.h>
#include <opencv2/objdetect/objdetect.hpp>

namespace dip {

class FaceMasker : public cv::CascadeClassifier::MaskGenerator {
public:
  FaceMasker() : valid_mask_(NULL), head_mask_(NULL), depth_integral_(NULL),
                 valid_integral_(NULL), head_integral_(NULL),
                 min_sizes_(NULL), max_sizes_(NULL),
                 size_(0), frame_(0), scale_(0) {}
  ~FaceMasker();

  void Run(int min_depth, int min_pixels, int open_size,
           int head_width, int head_height, int head_depth,
           int face_size, int extended_size, int window_size,
           int width, int height, float focal_length,
           const Depth *depth, Color *color);

  // MaskGenerator functions.
  cv::Mat generateMask(const cv::Mat& src);
  void initializeMask(const cv::Mat& src) {}

private:
  void Integral(int width, int height, bool *valid, const Depth *depth,
                int *integral);
  void Integral(int width, int height, bool flag, const bool *mask,
                int *integral);

  void Erode(int width, int height, int half_window, const int *integral,
             bool *mask);
  void Dilate(int width, int height, int half_window, const int *integral,
              bool *mask);

  int Sum(int width, int height, int left, int right, int top, int bottom,
          const int *integral);
  int Mean(int width, int height, int left, int right, int top, int bottom,
           const int *value_integral, const int *valid_integral);

  bool *valid_mask_, *head_mask_;
  int *depth_integral_, *valid_integral_, *head_integral_;

  float *min_sizes_, *max_sizes_;

  int window_size_;
  int width_, height_, size_;
  int frame_, scale_;

  DISALLOW_COPY_AND_ASSIGN(FaceMasker);
};

} // namespace dip

#endif // DIP_SEGMENTATION_FACEMASKER_H
