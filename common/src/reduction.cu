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
#include <dip/common/memory.h>

#define BLOCK_SIZE 256

namespace dip {

__global__ void Reduction(int elements, int *buffer) {
  // Allocate Shared Memory
  __shared__ int sm[2 * BLOCK_SIZE];

  // Get Block and Thread Id
  int b = blockIdx.x;
  int t = threadIdx.x;

  // Cooperative Load Elements into Shared Memory
  if ((t + (b * BLOCK_SIZE * 2)) < elements)
    sm[t] = buffer[t + (b * BLOCK_SIZE * 2)];
  else
    sm[t] = 0;

  if (((t + BLOCK_SIZE) + (b * BLOCK_SIZE * 2)) < elements)
    sm[t + BLOCK_SIZE] = buffer[(t + BLOCK_SIZE) + (b * BLOCK_SIZE * 2)];
  else
    sm[t + BLOCK_SIZE] = 0;

  // Initialize Offset
  int offset = 1;

  // Perform Reduction
  for (int d = (2 * BLOCK_SIZE) >> 1; d > 0; d >>= 1) {
    __syncthreads();

    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;

      sm[bi] += sm[ai];
    }

    offset *= 2;
  }

  // Store Results
  if (t == 0)
    buffer[b] = sm[(2 * BLOCK_SIZE) - 1];
}

__global__ void Reduction(int elements, float *buffer) {
  // Allocate Shared Memory
  __shared__ float sm[2 * BLOCK_SIZE];

  // Get Block and Thread Id
  int b = blockIdx.x;
  int t = threadIdx.x;

  // Cooperative Load Elements into Shared Memory
  if ((t + (b * BLOCK_SIZE * 2)) < elements)
    sm[t] = buffer[t + (b * BLOCK_SIZE * 2)];
  else
    sm[t] = 0;

  if (((t + BLOCK_SIZE) + (b * BLOCK_SIZE * 2)) < elements)
    sm[t + BLOCK_SIZE] = buffer[(t + BLOCK_SIZE) + (b * BLOCK_SIZE * 2)];
  else
    sm[t + BLOCK_SIZE] = 0;

  // Initialize Offset
  int offset = 1;

  // Perform Reduction
  for (int d = (2 * BLOCK_SIZE) >> 1; d > 0; d >>= 1) {
    __syncthreads();

    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;

      sm[bi] += sm[ai];
    }

    offset *= 2;
  }

  // Store Results
  if (t == 0)
    buffer[b] = sm[(2 * BLOCK_SIZE) - 1];
}

int Reduce(int elements, int *buffer) {
  // Launch Reduction Kernel
  while(elements > 1) {
    int grid_size = (elements + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);

    dim3 grid_dim(grid_size, 1, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);

    Reduction<<<grid_dim, block_dim>>>(elements, buffer);

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    elements = grid_size;
  }

  int value;
  Download(&value, buffer, sizeof(int));
  return value;
}

float Reduce(int elements, float *buffer) {
  // Launch Reduction Kernel
  while(elements > 1) {
    int grid_size = (elements + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);

    dim3 grid_dim(grid_size, 1, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);

    Reduction<<<grid_dim, block_dim>>>(elements, buffer);

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    elements = grid_size;
  }

  float value;
  Download(&value, buffer, sizeof(float));
  return value;
}

} // namespace dip
