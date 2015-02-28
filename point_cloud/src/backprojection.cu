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

__global__ void ComputeVertices(int width, int height, float fx, float fy,
                                float cx, float cy, const Depth *depth,
                                Vertices vertices) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  // Compute Vertices
  if ((col < width) && (row < height)) {
    int i = col + row * width;

    Depth depth_value = depth[i];

    if (depth_value > 0) {
      vertices.x[i] = depth_value * ((col - cx) / fx);
      vertices.y[i] = depth_value * ((row - cy) / fy);
      vertices.z[i] = depth_value;
    }
    else {
      vertices.x[i] = 0.0f;
      vertices.y[i] = 0.0f;
      vertices.z[i] = 0.0f;
    }
  }
}

__global__ void ComputeNormals(int width, int height, Vertices vertices,
                               Normals normals) {
  // Allocate Shared Memory
  __shared__ Vertex vs[BLOCK_WIDTH][BLOCK_WIDTH];

  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  // Cooperative Load of the Tile
  if ((col < width) && (row < height)) {
    vs[ty][tx].x = vertices.x[col + row * width];
    vs[ty][tx].y = vertices.y[col + row * width];
    vs[ty][tx].z = vertices.z[col + row * width];
  }
  else {
    vs[ty][tx].x = 0.0f;
    vs[ty][tx].y = 0.0f;
    vs[ty][tx].z = 0.0f;
  }

  // Sync Threads in Block
  __syncthreads();

  // Compute Normals
  if ((col < width) && (row < height)) {
    int i = col + row * width;

    // Load Center Vertex
    Vertex center;
    center.x = vs[ty][tx].x;
    center.y = vs[ty][tx].y;
    center.z = vs[ty][tx].z;

    // Load Neighboring Vertices
    Vertex west, north;
    if ((col - 1) >= 0) {
      if ((tx - 1) >= 0) {
        west.x = vs[ty][tx - 1].x;
        west.y = vs[ty][tx - 1].y;
        west.z = vs[ty][tx - 1].z;
      }
      else {
        west.x = vertices.x[i - 1];
        west.y = vertices.y[i - 1];
        west.z = vertices.z[i - 1];
      }
    }
    else {
      west.x = 0.0f;
      west.y = 0.0f;
      west.z = 0.0f;
    }

    if ((row - 1) >= 0) {
      if ((ty - 1) >= 0) {
        north.x = vs[ty - 1][tx].x;
        north.y = vs[ty - 1][tx].y;
        north.z = vs[ty - 1][tx].z;
      }
      else {
        north.x = vertices.x[i - width];
        north.y = vertices.y[i - width];
        north.z = vertices.z[i - width];
      }
    }
    else {
      north.x = 0.0f;
      north.y = 0.0f;
      north.z = 0.0f;
    }

    if((center.z > 0.0f) && (west.z > 0.0f) && (north.z > 0.0f)) {
      // Compute Vectors
      Vector left, up;
      left.x = west.x - center.x;
      left.y = west.y - center.y;
      left.z = west.z - center.z;

      up.x = north.x - center.x;
      up.y = north.y - center.y;
      up.z = north.z - center.z;

      // Perform Cross Product
      Vector normal;
      normal.x = left.y * up.z - up.y * left.z;
      normal.y = up.x * left.z - left.x * up.z;
      normal.z = left.x * up.y - up.x * left.y;

      // Normalize
      float inorm = rsqrt(normal.x * normal.x + normal.y * normal.y +
                          normal.z * normal.z);

      normals.x[i] = normal.x * inorm;
      normals.y[i] = normal.y * inorm;
      normals.z[i] = normal.z * inorm;
    }
    else {
      normals.x[i] = 0.0f;
      normals.y[i] = 0.0f;
      normals.z[i] = 0.0f;
    }
  }
}

void BackProjectionKernel(int width, int height, float fx, float fy,
                          float cx, float cy, const Depth *depth,
                          Vertices vertices, Normals normals) {
  // Launch Back Projection Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  ComputeVertices<<<grid_dim, block_dim>>>(width, height, fx, fy, cx, cy, depth,
                                           vertices);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  ComputeNormals<<<grid_dim, block_dim>>>(width, height, vertices, normals);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
