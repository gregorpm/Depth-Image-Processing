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

__global__ void BilateralFilter(float sigma_d, float sigma_r,
                                int width, int height,
                                const Depth *depth, Depth *filtered_depth) {
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
  }
  else {
    ds[ty][tx] = 0;
  }

  // Sync Threads in Block
  __syncthreads();

  // Perform the Bilateral Filter
  if ((col < width) && (row < height)) {
    float center_depth = ds[ty][tx];
    float h = 0.0f, k = 0.0f;

    if (center_depth > 0) {
      for (int dy = -FILTER_HALF_WIDTH; dy <= FILTER_HALF_WIDTH; dy++) {
        for (int dx = -FILTER_HALF_WIDTH; dx <= FILTER_HALF_WIDTH; dx++) {
          int x = col + dx;
          int y = row + dy;

          if ((x >= 0) && (x < width) && (y >= 0) && (y < height)) {
            int i = tx + dx;
            int j = ty + dy;

            float current_depth;
            if ((i >= 0) && (i < BLOCK_WIDTH) && (j >= 0) && (j < BLOCK_WIDTH))
              current_depth = ds[j][i];
            else
              current_depth = depth[x + y * width];

            if (current_depth > 0) {
              float d = static_cast<float>((dx * dx) + (dy * dy));
              float r = static_cast<float>((current_depth - center_depth) *
                                           (current_depth - center_depth));

              float weight = __expf(-0.5f * (d * sigma_d + r * sigma_r));

              h += current_depth * weight;
              k += weight;
            }
          }
        }
      }
    }

    if (k > 0.0f)
      filtered_depth[col + row * width] = h / k;
    else
      filtered_depth[col + row * width] = 0;
  }
}

void BilateralKernel(float sigma_d, float sigma_r, int width, int height,
                     const Depth *depth, Depth *filtered_depth) {
  // Launch Bilateral Filter Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  BilateralFilter<<<grid_dim, block_dim>>>(sigma_d, sigma_r, width, height,
                                           depth, filtered_depth);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
