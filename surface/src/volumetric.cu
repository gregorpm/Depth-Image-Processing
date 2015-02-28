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
#include <dip/common/macros.h>
#include <dip/common/types.h>
#include <dip/surface/voxel.h>

#define BLOCK_WIDTH 16

namespace dip {

__constant__ float T[4][4];

__global__ void Integration(int volume_size, float volume_dimension,
                            float voxel_dimension, float max_truncation,
                            float max_weight, int width, int height,
                            float fx, float fy, float cx, float cy,
                            Vertex center, const Depth *depth,
                            const Normals normals, Voxel *volume) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int x = tx + bx * BLOCK_WIDTH;
  int y = ty + by * BLOCK_WIDTH;

  // Calculate volume index and step size
  int i = x + y * volume_size;
  int step = volume_size * volume_size;

  if ((x < volume_size) && (y < volume_size)) {
    // Convert from Gird Position to Global Position
    Vertex global;
    global.x = (x * voxel_dimension) - (volume_dimension / 2.0f) + center.x;
    global.y = (y * voxel_dimension) - (volume_dimension / 2.0f) + center.y;

    for (int z = 0; z < volume_size; z++, i += step) {
      global.z = (z * voxel_dimension) - (volume_dimension / 2.0f) + center.z;

      // Convert from Global Coordinate Space
      // to Camera Coordinate Space
      Vertex camera;
      camera.x = T[0][0] * global.x +
                 T[0][1] * global.y +
                 T[0][2] * global.z +
                 T[0][3];
      camera.y = T[1][0] * global.x +
                 T[1][1] * global.y +
                 T[1][2] * global.z +
                 T[1][3];
      camera.z = T[2][0] * global.x +
                 T[2][1] * global.y +
                 T[2][2] * global.z +
                 T[2][3];

      // Check if Vertex is in front of the camera
      if (camera.z > 0.0f) {
        // Perspective Project Vertex
        int u = (int)roundf((fx * camera.x + cx * camera.z) / camera.z);
        int v = (int)roundf((fy * camera.y + cy * camera.z) / camera.z);

        // Check Pixel Position
        if ((u >= 0 && u < width) && (v >= 0 && v < height)) {
          int j = u + v * width;

          // Load Depth Value
          Depth depth_value = depth[j];

          if (depth_value > 0) {
            float sdf, tsdf;

#ifdef DISTANCE_OPTICAL_AXIS
            // Distance to surface along optical axis.
            sdf = depth_value - camera.z;
#else
            // Distance to surface along camera ray.
            float voxel_distance = sqrt(camera.x * camera.x +
                                        camera.y * camera.y +
                                        camera.z * camera.z);

            Vector ray;
            ray.x = (u - cx) / fx;
            ray.y = (v - cy) / fy;
            ray.z = 1.0f;

            float surface_distance = depth_value * sqrt(ray.x * ray.x +
                                                        ray.y * ray.y +
                                                        ray.z * ray.z);

            sdf = surface_distance - voxel_distance;
#endif

            if (sdf >= -max_truncation) {
              if (sdf > 0)
                tsdf = MIN(1.0f, sdf / max_truncation);
              else
                tsdf = MAX(-1.0f, sdf / max_truncation);

              // Cosine angle between view direction and the surface normal.
              float cosine_angle = normals.z[j];

              // Compute Weight
              float weight = (exp(cosine_angle - 1.0f) *
                              exp(-0.0001f * depth_value)) / max_weight;

              // Update Voxel
              Voxel voxel = volume[i];

              float f = UNCOMPRESS_VALUE(voxel);
              float w = UNCOMPRESS_WEIGHT(voxel);

              if(w != 0) {
                f = (f * w + tsdf * weight) / (w + weight);
                w = MIN(w + weight, 1.0f);
              }
              else {
                f = tsdf;
                w = weight;
              }

              volume[i] = COMPRESS(f, w);
            }
          }
        }
      }
    }
  }
}

void VolumetricKernel(int volume_size, float volume_dimension,
                      float voxel_dimension, float max_truncation,
                      float max_weight, int width, int height,
                      float fx, float fy, float cx, float cy,
                      Vertex center, float *transformation,
                      const Depth *depth, const Normals normals,
                      Voxel *volume) {
  // Copy Transforms to Constant Memory
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(T, transformation, sizeof(float) * 16));

  // Launch Volumetric Kernel
  int grid_width = (volume_size + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (volume_size + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  Integration<<<grid_dim, block_dim>>>(volume_size, volume_dimension,
                                       voxel_dimension, max_truncation,
                                       max_weight, width, height, fx, fy,
                                       cx, cy, center, depth, normals, volume);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
