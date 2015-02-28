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

__global__ void ThresholdFilter(int min_depth, int max_depth,
                                int width, int height, Depth *depth) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  // Perform Threshold
  if ((col < width) && (row < height)) {
    int i = col + row * width;

    int depth_value = depth[i];

    if((depth_value < min_depth) || (depth_value > max_depth))
      depth[i] = 0;
  }
}

void ThresholdKernel(int min_depth, int max_depth, int width, int height,
                     Depth *depth) {
  // Launch Threshold Filter Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  ThresholdFilter<<<grid_dim, block_dim>>>(min_depth, max_depth, width, height,
                                           depth);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
