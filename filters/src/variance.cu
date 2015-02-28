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

#include <dip/common/error.h>
#include <dip/common/types.h>

#define FILTER_HALF_WIDTH 3
#define BLOCK_WIDTH       16

namespace dip {

__global__ void Variance(int width, int height, const Depth *depth,
                         float *variance, float *std, float *valid) {
  // Allocate Shared Memory
  __shared__ Depth ds[BLOCK_WIDTH][BLOCK_WIDTH];

  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  // Cooperative Load of the Tile
  if ((col < width) && (row < height)) {
    ds[ty][tx] = depth[col + row * width];
  } else {
    ds[ty][tx] = 0;
  }

  // Sync Threads in Block
  __syncthreads();

  // Perform the Variance Filter
  if ((col < width) && (row < height)) {
    float sum = 0.0f, squared_sum = 0.0f;
    int count = 0;

    for (int dy = -FILTER_HALF_WIDTH; dy <= FILTER_HALF_WIDTH; dy++) {
      for (int dx = -FILTER_HALF_WIDTH; dx <= FILTER_HALF_WIDTH; dx++) {
        int x = col + dx;
        int y = row + dy;

        if ((x >= 0) && (x < width) && (y >= 0) && (y < height)) {
          int i = tx + dx;
          int j = ty + dy;

          float depth_value;
          if ((i >= 0) && (i < BLOCK_WIDTH) && (j >= 0) && (j < BLOCK_WIDTH))
            depth_value = ds[j][i];
          else
            depth_value = depth[x + y * width];

          if (depth_value > 0) {
            sum += depth_value;
            squared_sum += depth_value * depth_value;
            count++;
          }
        }
      }
    }

    if ((ds[ty][tx] > 0) && (count > 0)) {
      float mean = sum / count;
      float squared_mean = squared_sum / count;
      float var = squared_mean - (mean * mean);

      if (var > 0.0f) {
        variance[col + row * width] = var;
        std[col + row * width] = sqrt(var);
        valid[col + row * width] = 1.0f;
      } else {
        variance[col + row * width] = 0.0f;
        std[col + row * width] = 0.0f;
        valid[col + row * width] = 0.0f;
      }
    } else {
      variance[col + row * width] = 0.0f;
      std[col + row * width] = 0.0f;
      valid[col + row * width] = 0.0f;
    }
  }
}

__global__ void Threshold(float threshold, int width, int height,
                          const float *std, const Depth *depth,
                          Depth *filtered_depth) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  // Perform Threshold
  if ((col < width) && (row < height)) {
    int i = col + row * width;

    if (std[i] < threshold)
      filtered_depth[i] = depth[i];
    else
      filtered_depth[i] = 0;
  }
}

void VarianceKernel(int width, int height, const Depth *depth,
                    float *variance, float *std, float *valid) {
  // Launch Variance Filter Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  Variance<<<grid_dim, block_dim>>>(width, height, depth, variance, std, valid);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void ThresholdKernel(float threshold, int width, int height,
                     const float *std, const Depth *depth,
                     Depth *filtered_depth) {
  // Launch Variance Filter Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  Threshold<<<grid_dim, block_dim>>>(threshold, width, height, std,
                                     depth, filtered_depth);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
