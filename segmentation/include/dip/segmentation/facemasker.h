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

#ifndef DIP_SEGMENTATION_FACEMASKER_H
#define DIP_SEGMENTATION_FACEMASKER_H

#include <dip/common/distance.h>
#include <dip/common/types.h>
#include <dip/common/macros.h>
#include <opencv2/objdetect/objdetect.hpp>

namespace dip {

class FaceMasker : public cv::CascadeClassifier::MaskGenerator {
public:
  FaceMasker() : boundary_(NULL), distances_(NULL),
                 min_sizes_(NULL), max_sizes_(NULL), size_(0) {}
  ~FaceMasker();

  void Run(int max_difference, int min_depth, int max_depth,
           float min_face_size, float max_face_size, int window_size,
           int width, int height, float focal_length, const Depth *depth);

  // MaskGenerator functions.
  cv::Mat generateMask(const cv::Mat& src);
  void initializeMask(const cv::Mat& src) {}

private:
  Distance distance_;

  bool *boundary_;
  unsigned int *distances_;

  float *min_sizes_;
  float *max_sizes_;

  int window_size_;
  int width_, height_, size_;

  DISALLOW_COPY_AND_ASSIGN(FaceMasker);
};

} // namespace dip

#endif // DIP_SEGMENTATION_FACEMASKER_H
