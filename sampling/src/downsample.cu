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

#define BLOCK_WIDTH 16

namespace dip {

__global__ void DownsampleDepth(int factor, int max_difference,
                                int width, int height,
                                int downsampled_width, int downsampled_height,
                                const Depth *depth, Depth *downsampled_depth) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int output_col = tx + bx * BLOCK_WIDTH;
  int output_row = ty + by * BLOCK_WIDTH;
  int input_col = output_col << factor;
  int input_row = output_row << factor;

  // Perform block average and downsample
  if ((output_col < downsampled_width) && (output_row < downsampled_height)) {
    if ((input_col < width) && (input_row < height)) {
      float center_depth = depth[input_col + input_row * width];

      // Block average on input depth image
      int size = 1 << factor;
      float h = 0.0f, k = 0.0f;

      for (int dy = 0; dy < size; dy++) {
        for (int dx = 0; dx < size; dx++) {
          int x = input_col + dx;
          int y = input_row + dy;

          if ((x < width) && (y < height)) {
            float current_depth = depth[x + y * width];
            float difference = fabs(current_depth - center_depth);

            if(difference < max_difference) {
              h += current_depth;
              k++;
            }
          }
        }
      }

      // Downsample depth image
      downsampled_depth[output_col + output_row * downsampled_width] = h / k;
    }
  }
}

void DownsampleKernel(int factor, int max_difference,
                      int width, int height,
                      int downsampled_width, int downsampled_height,
                      const Depth *depth, Depth *downsampled_depth) {
  // Launch Downsample Kernel
  int grid_width = (downsampled_width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (downsampled_height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  DownsampleDepth<<<grid_dim, block_dim>>>(factor, max_difference,
                                           width, height,
                                           downsampled_width,
                                           downsampled_height,
                                           depth, downsampled_depth);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
