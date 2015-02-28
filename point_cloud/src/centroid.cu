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

__constant__ float *buffer_ptr[4];

__global__ void Initialize(int width, int height, Vertices vertices) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  if ((col < width) && (row < height)) {
    int i = col + row * width;

    Vertex vertex;
    vertex.x = vertices.x[i];
    vertex.y = vertices.y[i];
    vertex.z = vertices.z[i];

    if (vertex.z > 0.0f) {
      buffer_ptr[0][i] = vertex.x;
      buffer_ptr[1][i] = vertex.y;
      buffer_ptr[2][i] = vertex.z;
      buffer_ptr[3][i] = 1.0f;
    }
    else {
      buffer_ptr[0][i] = 0.0f;
      buffer_ptr[1][i] = 0.0f;
      buffer_ptr[2][i] = 0.0f;
      buffer_ptr[3][i] = 0.0f;
    }
  }
}

void CentroidKernel(int width, int height, Vertices vertices,
                    float *buffer[4]) {
  // Copy buffer pointers to Constant Memory
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(buffer_ptr, buffer, sizeof(float*) * 4));

  // Launch Centroid Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  Initialize<<<grid_dim, block_dim>>>(width, height, vertices);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
