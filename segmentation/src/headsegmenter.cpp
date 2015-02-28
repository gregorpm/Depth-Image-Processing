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

#include <dip/segmentation/headsegmenter.h>

#include <string.h>

namespace dip {

HeadSegmenter::~HeadSegmenter() {
  if (labels_ != NULL)
    delete [] labels_;
  if (column_histogram_ != NULL)
    delete [] column_histogram_;
  if (row_histogram_ != NULL)
    delete [] row_histogram_;
}

int HeadSegmenter::Run(int min_depth, int max_depth, int max_difference,
                       int min_width, int min_height,
                       int max_width, int max_height,
                       float fx, float fy, int width, int height,
                       const Depth *depth, Depth *segmented_depth) {
  if (size_ < (width * height)) {
    size_ = width * height;

    if (labels_ != NULL)
      delete [] labels_;

    labels_ = new int[size_];
  }

  if (width_ < width) {
    width_ = width;

    if (column_histogram_ != NULL)
      delete [] column_histogram_;

    column_histogram_ = new int[width_];
  }

  if (height_ < height) {
    height_ = height;

    if (row_histogram_ != NULL)
      delete [] row_histogram_;

    row_histogram_ = new int[height_];
  }

  memset(segmented_depth, 0, sizeof(Depth) * width * height);

  // Determine foreground region
  connected_components_.Run(max_difference, width, height, depth, components_,
                            labels_);

  int foreground_id = 0;
  int foreground_size = 0;
  int foreground_depth = 0;

  for (unsigned int n = 0; n < components_.size(); n++) {
    if (components_.at(n).root) {
      if (components_.at(n).mean > min_depth) {
        if (components_.at(n).mean < max_depth) {
          if (components_.at(n).size > foreground_size) {
            foreground_id = components_.at(n).parent;
            foreground_size = components_.at(n).size;
            foreground_depth = components_.at(n).mean;
          }
        }
      }
    }
  }

  // Determine head region
  memset(column_histogram_, 0, sizeof(int) * width);
  memset(row_histogram_, 0, sizeof(int) * height);

  // Generate Histograms
  int count = 0;
  int i = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++, i++) {
      if (labels_[i] == foreground_id) {
        column_histogram_[x]++;
        row_histogram_[y]++;

        count++;
      }
    }
  }

  // Locate Top of Head
  int max_value = 0;
  int head_center = 0;
  for (int x = 0; x < width; x++) {
    if (column_histogram_[x] > max_value) {
      max_value = column_histogram_[x];
      head_center = x;
    }
  }

  int head_top = 0;
  for (int y = 0; y < height; y++) {
    int i = head_center + y * width;

    if (labels_[i] == foreground_id) {
      head_top = y;
      break;
    }
  }

  // Locate Bottom of Head
  float max_variance = 0.0f;

  float weight_torso = 0.0f, weight_head = 0.0f;
  int count_head = 0;

  int head_bottom = 0;
  for (int y = head_top; y < height; y++) {
    // Compute Weights
    weight_head++;
    weight_torso = (float)((height - 1) - head_top) - weight_head;

    if ((weight_head > 0) && (weight_torso > 0)) {
      count_head += row_histogram_[y];

      // Compute Means
      float mean_torso, mean_head;
      mean_head = count_head / weight_head;
      mean_torso = (count - count_head) / weight_torso;

      // Compute Between Class Variance
      float between_variance;
      between_variance = (weight_head / ((height - 1) - head_top)) *
                         (weight_torso / ((height - 1) - head_top)) *
                         (mean_head - mean_torso) * (mean_head - mean_torso);

      if (between_variance > max_variance) {
        max_variance = between_variance;
        head_bottom = y;
      }
    }
  }

  head_bottom -= (int)((head_bottom - head_top) * 0.10f);

  // Locate Left and Right side of Head
  int head_left = width;
  int head_right = 0;
  for (int y = head_top; y < head_bottom; y++) {
    for (int x = 0; x < width; x++) {
      int i = x + y * width;

      if (labels_[i] == foreground_id) {
        if (x < head_left)
          head_left = x;

        if (x > head_right)
          head_right = x;
      }
    }
  }

  // Check head dimensions.
  float head_width = ((head_right - head_left) * foreground_depth) / fx;
  float head_height = ((head_bottom - head_top) * foreground_depth) / fy;

  if ((head_width > min_width) && (head_width < max_width)) {
    if ((head_height > min_height) && (head_height < max_height)) {
      // Segment User's Head
      for (int y = head_top; y < head_bottom; y++) {
        for (int x = head_left; x < head_right; x++) {
          int i = x + y * width;

          if (labels_[i] == foreground_id)
            segmented_depth[i] = depth[i];
        }
      }

      return 0;
    }
  }

  return -1;
}

} // namespace dip
