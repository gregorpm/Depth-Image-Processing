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

#include <dip/segmentation/facemasker.h>

#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace dip {

FaceMasker::~FaceMasker() {
  if (valid_mask_ != NULL)
    delete [] valid_mask_;
  if (head_mask_ != NULL)
    delete [] head_mask_;
  if (depth_integral_ != NULL)
    delete [] depth_integral_;
  if (valid_integral_ != NULL)
    delete [] valid_integral_;
  if (head_integral_ != NULL)
    delete [] head_integral_;
  if (min_sizes_ != NULL)
    delete [] min_sizes_;
  if (max_sizes_ != NULL)
    delete [] max_sizes_;
}

void FaceMasker::Run(int min_depth, int min_pixels, int open_size,
                     int head_width, int head_height, int head_depth,
                     int face_size, int extended_size,
                     int window_size, int width, int height,
                     float focal_length, const Depth *depth,
                     Color *color) {
  width_ = width;
  height_ = height;
  window_size_ = window_size;

  if (size_ < (width_ * height_)) {
    size_ = width_ * height_;

    if (valid_mask_ != NULL)
      delete [] valid_mask_;
    if (head_mask_ != NULL)
      delete [] head_mask_;
    if (depth_integral_ != NULL)
      delete [] depth_integral_;
    if (valid_integral_ != NULL)
      delete [] valid_integral_;
    if (head_integral_ != NULL)
      delete [] head_integral_;
    if (min_sizes_ != NULL)
      delete [] min_sizes_;
    if (max_sizes_ != NULL)
      delete [] max_sizes_;

    valid_mask_ = new bool[size_];
    head_mask_ = new bool[size_];
    depth_integral_ = new int[size_];
    valid_integral_ = new int[size_];
    head_integral_ = new int[size_];
    min_sizes_ = new float[size_];
    max_sizes_ = new float[size_];
  }

  int max_depth = (head_width * focal_length) / min_pixels;

  #pragma omp parallel for
  for (int i = 0; i < (width * height); i++) {
    valid_mask_[i] = ((depth[i] > min_depth) && (depth[i] < max_depth)) ?
                     true : false;
  }

  Integral(width, height, true, valid_mask_, valid_integral_);
  Integral(width, height, valid_mask_, depth, depth_integral_);

  #pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = x + y * width;

      head_mask_[i] = false;
      if (valid_mask_[i]) {
        int head_cols = (int)((head_width * focal_length) / depth[i]);
        int head_rows = (int)((head_height * focal_length) / depth[i]);

        int center_average = Mean(width, height,
            x - head_cols / 2, x + head_cols / 2,
            y - head_rows / 2, y + head_rows / 2,
            depth_integral_, valid_integral_);

        int left_average = Mean(width, height,
            x - (5 * head_cols / 4), x - (3 * head_cols / 4),
            y - head_rows / 2, y + head_rows / 2,
            depth_integral_, valid_integral_);

        int right_average = Mean(width, height,
            x + (3 * head_cols / 4), x + (5 * head_cols / 4),
            y - head_rows / 2, y + head_rows / 2,
            depth_integral_, valid_integral_);

        int top_average = Mean(width, height,
            x - head_cols / 2, x + head_cols / 2,
            y - (5 * head_rows / 4), y - (3 * head_rows / 4),
            depth_integral_, valid_integral_);

        int top_left_average = Mean(width, height,
            x - (5 * head_cols / 4), x - (3 * head_cols / 4),
            y - (5 * head_rows / 4), y - (3 * head_rows / 4),
            depth_integral_, valid_integral_);

        int top_right_average = Mean(width, height,
            x + (3 * head_cols / 4), x + (5 * head_cols / 4),
            y - (5 * head_rows / 4), y - (3 * head_rows / 4),
            depth_integral_, valid_integral_);

        int center_difference = ABS(depth[i] - center_average);
        int left_difference = ABS(depth[i] - left_average);
        int right_difference = ABS(depth[i] - right_average);
        int top_difference = ABS(depth[i] - top_average);
        int top_left_difference = ABS(depth[i] - top_left_average);
        int top_right_difference = ABS(depth[i] - top_right_average);

        int alpha = head_depth;
        int beta = 2 * head_depth;
        head_mask_[i] = ((center_difference < alpha) &&
                         (left_difference > beta) &&
                         (right_difference > beta) &&
                         (top_difference > beta) &&
                         (top_left_difference > beta) &&
                         (top_right_difference > beta)) ? true : false;
      }
    }
  }

  Integral(width, height, false, head_mask_, head_integral_);
  Erode(width, height, open_size, head_integral_, head_mask_);
  Integral(width, height, true, head_mask_, head_integral_);
  Dilate(width, height, open_size, head_integral_, head_mask_);

  Integral(width, height, true, head_mask_, head_integral_);

  #pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = x + y * width;

      min_sizes_[i] = max_sizes_[i] = 0.0f;
      if (valid_mask_[i]) {
        int face_pixels = (int)((face_size * focal_length) / depth[i]);

        if (Sum(width, height, x - face_pixels / 2, x + face_pixels / 2,
                y - face_pixels / 2, y + face_pixels / 2, head_integral_) > 0) {
          int extended_pixels =(int)((extended_size * focal_length) / depth[i]);

          min_sizes_[i] = face_pixels - extended_pixels;
          max_sizes_[i] = face_pixels + extended_pixels;
        }
      }
    }
  }

  frame_++;
  scale_ = 0;

//#define HEAD_DEBUG
#ifdef HEAD_DEBUG
  Mat image(height_, width_, CV_8UC3, color);
  Mat head_region = Mat::zeros(height_, width_, CV_8UC3);

  #pragma omp parallel for
  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      int i = x + y * width_;

      if (head_mask_[i])
        head_region.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
    }
  }

  char filename[256];
  sprintf(filename, "head-%d.png", frame_);

  Mat output;
  addWeighted(image, 0.5, head_region, 0.5, 0.0, output);

  imwrite(filename, output);
  scale_++;
#endif
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

//#define MASK_DEBUG
#ifdef MASK_DEBUG
  Mat shifted_mask = Mat::zeros(src.size(), CV_8U);

  #pragma omp parallel for
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      int Y = (int)(y * inv_scale);
      int X = (int)(x * inv_scale);

      if ((Y < height_) && (X < width_)) {
        int i = X + Y * width_;

        if ((scaled_window_size >= min_sizes_[i]) &&
            (scaled_window_size <= max_sizes_[i])) {
          shifted_mask.at<unsigned char>(y, x) = 255;
        }
      }
    }
  }

  char filename[256];
  sprintf(filename, "mask-%d-%d.png", frame_, scale_);

  Mat output;
  addWeighted(src, 0.5, shifted_mask, 0.5, 0.0, output);
  rectangle(output, Point(0, 0), Point(window_size_, window_size_),
            Scalar(255));

  imwrite(filename, output);
  scale_++;
#endif

  return mask;
}

void FaceMasker::Integral(int width, int height, bool *valid,
                          const Depth *depth, int *integral) {
  int i = 0;
  for (int y = 0; y < height; y++) {
    int sum = 0;
    for (int x = 0; x < width; x++, i++) {
      if (valid[i])
        sum += depth[i];

      integral[i] = sum + ((y > 0) ? integral[i - width] : 0);
    }
  }
}

void FaceMasker::Integral(int width, int height, bool flag, const bool *mask,
                          int *integral) {
  int i = 0;
  for (int y = 0; y < height; y++) {
    int sum = 0;
    for (int x = 0; x < width; x++, i++) {
      sum += (mask[i] == flag) ? 1 : 0;
      integral[i] = sum + ((y > 0) ? integral[i - width] : 0);
    }
  }
}

void FaceMasker::Erode(int width, int height, int half_window,
                       const int *integral, bool *mask) {
  #pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = x + y * width;

      if (mask[i]) {
        if (Sum(width, height, x - half_window, x + half_window,
                y - half_window, y + half_window, integral) > 0) {
          mask[i] = false;
        }
      }
    }
  }
}

void FaceMasker::Dilate(int width, int height, int half_window,
                        const int *integral, bool *mask) {
  #pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = x + y * width;

      if (!mask[i]) {
        if (Sum(width, height, x - half_window, x + half_window,
                y - half_window, y + half_window, integral) > 0) {
          mask[i] = true;
        }
      }
    }
  }
}

int FaceMasker::Sum(int width, int height, int left, int right,
                    int top, int bottom, const int *integral) {
  int x1 = MIN(MAX(left, 0), width - 1);
  int y1 = MIN(MAX(top, 0), height - 1);
  int x2 = MIN(MAX(right, 0), width -1);
  int y2 = MIN(MAX(bottom, 0), height - 1);

  int a = integral[x2 + y2 * width];
  int b = (y1 > 0) ? integral[x2 + (y1 - 1) * width] : 0;
  int c = (x1 > 0) ? integral[(x1 - 1) + y2 * width] : 0;
  int d = ((x1 > 0) && (y1 > 0)) ? integral[(x1 - 1) + (y1 - 1) * width] : 0;

  return (a - b - c + d);
}

int FaceMasker::Mean(int width, int height, int left, int right,
                     int top, int bottom, const int *value_integral,
                     const int *valid_integral) {
  int x1 = MIN(MAX(left, 0), width - 1);
  int y1 = MIN(MAX(top, 0), height - 1);
  int x2 = MIN(MAX(right, 0), width -1);
  int y2 = MIN(MAX(bottom, 0), height - 1);

  int a, b, c, d;

  a = valid_integral[x2 + y2 * width];
  b = (y1 > 0) ? valid_integral[x2 + (y1 - 1) * width] : 0;
  c = (x1 > 0) ? valid_integral[(x1 - 1) + y2 * width] : 0;
  d = ((x1 > 0) && (y1 > 0)) ? valid_integral[(x1 - 1) + (y1 - 1) * width] : 0;

  int size = (a - b - c + d);
  int window_size = (x2 - x1 + 1) * (y2 - y1 + 1);

  if (size > (window_size / 4)) {
    a = value_integral[x2 + y2 * width];
    b = (y1 > 0) ? value_integral[x2 + (y1 - 1) * width] : 0;
    c = (x1 > 0) ? value_integral[(x1 - 1) + y2 * width] : 0;
    d = ((x1 > 0) && (y1 > 0)) ? value_integral[(x1 - 1) + (y1 - 1) * width] : 0;

    return (a - b - c + d) / size;
  }

  return 0;
}

} // namespace dip
