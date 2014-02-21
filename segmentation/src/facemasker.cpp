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

#include <dip/segmentation/facemasker.h>

using namespace cv;

namespace dip {

FaceMasker::~FaceMasker() {
  if (boundary_ != NULL)
    delete [] boundary_;
  if (distances_ != NULL)
    delete [] distances_;
  if (min_sizes_ != NULL)
    delete [] min_sizes_;
  if (max_sizes_ != NULL)
    delete [] max_sizes_;
}

void FaceMasker::Run(int max_difference, int min_depth, int max_depth,
                     float min_face_size, float max_face_size,
                     int window_size, int width, int height,
                     float focal_length, const Depth *depth) {
  width_ = width;
  height_ = height;
  window_size_ = window_size;

  if (size_ < (width_ * height_)) {
    size_ = width_ * height_;

    if (boundary_ != NULL)
      delete [] boundary_;
    if (distances_ != NULL)
      delete [] distances_;
    if (min_sizes_ != NULL)
      delete [] min_sizes_;
    if (max_sizes_ != NULL)
      delete [] max_sizes_;

    boundary_ = new bool[size_];
    distances_ = new unsigned int[size_];
    min_sizes_ = new float[size_];
    max_sizes_ = new float[size_];
  }

  memset(boundary_, 0, sizeof(bool) * size_);

  #pragma omp parallel for
  for (int y = 1; y < height_ - 1; y++) {
    for (int x = 1; x < width_ - 1; x++) {
      int i = x + y * width_;

      if ((DIFF(depth[i], depth[i - 1]) > max_difference) ||
          (DIFF(depth[i], depth[i - width]) > max_difference)) {
        boundary_[i] = true;
      }
    }
  }

  distance_.Run(width_, height_, boundary_, distances_);

  memset(min_sizes_, 0, sizeof(float) * size_);
  memset(max_sizes_, 0, sizeof(float) * size_);

  #pragma omp parallel for
  for (int y = 1; y < height_ - 1; y++) {
    for (int x = 1; x < width_ - 1; x++) {
      int i = x + y * width_;

      if ((depth[i] > min_depth) && (depth[i] < max_depth)) {
        float min_size = (min_face_size * focal_length) / depth[i];
        float max_size = (max_face_size * focal_length) / depth[i];

        int mean_radius = (int)((min_size + max_size) / 4.0f);
        int difference = (int)((max_size - min_size) / 2.0f);

        if (x > mean_radius) {
          if (distances_[i - mean_radius] > difference)
            continue;
        }

        if (x < (width_ - mean_radius)) {
          if (distances_[i + mean_radius] > difference)
            continue;
        }

        if (y > mean_radius) {
          if (distances_[i - mean_radius * width_] > difference)
            continue;
        }

        min_sizes_[i] = min_size;
        max_sizes_[i] = max_size;
      }
    }
  }
}

Mat FaceMasker::generateMask(const Mat& src) {
  Mat mask = Mat::zeros(src.size(), CV_8U);

  float scale = (float)src.cols / (float)width_;
  float inv_scale = 1.0f / scale;
  float half_window = window_size_ / 2.0f;
  float scaled_window_size = window_size_ * inv_scale;

  int rows = (int)(src.rows - half_window);
  int cols = (int)(src.cols - half_window);

  #pragma omp parallel for
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      int Y = (int)((y + half_window) * inv_scale);
      int X = (int)((x + half_window) * inv_scale);

      if ((Y < height_) && (X < width_)) {
        int i = X + Y * width_;

        if ((scaled_window_size >= min_sizes_[i]) &&
            (scaled_window_size <= max_sizes_[i])) {
          mask.at<unsigned char>(y, x) = 255;
        }
      }
    }
  }

  return mask;
}

} // namespace dip
