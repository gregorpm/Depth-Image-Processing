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

#ifndef DIP_SEGMENTATION_HEADSEGMENTER_H
#define DIP_SEGMENTATION_HEADSEGMENTER_H

#include <dip/common/macros.h>
#include <dip/common/types.h>
#include <dip/segmentation/connectedcomponents.h>

namespace dip {

class HeadSegmenter {
public:
  HeadSegmenter() : labels_(NULL), column_histogram_(NULL),
                    row_histogram_(NULL), width_(0), height_(0), size_(0) {}
  ~HeadSegmenter();

  int Run(int min_depth, int max_depth, int max_difference,
          int min_width, int min_height, int max_width, int max_height,
          float fx, float fy, int width, int height, const Depth *depth,
          Depth *segmented_depth);

private:
  ConnectedComponents connected_components_;
  std::vector<CC> components_;
  int *labels_;

  int *column_histogram_;
  int *row_histogram_;

  int width_, height_, size_;

  DISALLOW_COPY_AND_ASSIGN(HeadSegmenter);
};

} // namespace dip

#endif // DIP_SEGMENTATION_HEADSEGMENTER_H
