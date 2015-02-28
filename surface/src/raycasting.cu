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

__device__ bool interpolate(int volume_size, float volume_dimension,
                            float voxel_dimension, float min_weight,
                            const Voxel *volume, Vertex center,
                            float vx, float vy, float vz, float &tsdf) {
  // Convert Vertex Position to Grid Position
  float gx, gy, gz;
  gx = (vx + (volume_dimension / 2.0f) - center.x) / voxel_dimension;
  gy = (vy + (volume_dimension / 2.0f) - center.y) / voxel_dimension;
  gz = (vz + (volume_dimension / 2.0f) - center.z) / voxel_dimension;

  float gx0, gy0, gz0;
  float gx1, gy1, gz1;

  gx0 = floor(gx);
  gx1 = gx0 + 1.0f;
  gy0 = floor(gy);
  gy1 = gy0 + 1.0f;
  gz0 = floor(gz);
  gz1 = gz0 + 1.0f;

  int i;
  float f000, f001, f010, f011, f100, f101, f110, f111;
  float w000, w001, w010, w011, w100, w101, w110, w111;

  i = (int)gx0 + (int)gy0 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f000 = UNCOMPRESS_VALUE(volume[i]);
    w000 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f000 = 0.0f;
    w000 = 0.0f;
  }

  i = (int)gx0 + (int)gy0 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f001 = UNCOMPRESS_VALUE(volume[i]);
    w001 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f001 = 0.0f;
    w001 = 0.0f;
  }

  i = (int)gx0 + (int)gy1 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f010 = UNCOMPRESS_VALUE(volume[i]);
    w010 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f010 = 0.0f;
    w010 = 0.0f;
  }

  i = (int)gx0 + (int)gy1 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f011 = UNCOMPRESS_VALUE(volume[i]);
    w011 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f011 = 0.0f;
    w011 = 0.0f;
  }

  i = (int)gx1 + (int)gy0 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f100 = UNCOMPRESS_VALUE(volume[i]);
    w100 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f100 = 0.0f;
    w100 = 0.0f;
  }

  i = (int)gx1 + (int)gy0 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f101 = UNCOMPRESS_VALUE(volume[i]);
    w101 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f101 = 0.0f;
    w101 = 0.0f;
  }

  i = (int)gx1 + (int)gy1 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f110 = UNCOMPRESS_VALUE(volume[i]);
    w110 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f110 = 0.0f;
    w110 = 0.0f;
  }

  i = (int)gx1 + (int)gy1 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f111 = UNCOMPRESS_VALUE(volume[i]);
    w111 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f111 = 0.0f;
    w111 = 0.0f;
  }

  if((w000 <= min_weight) || (w001 <= min_weight) ||
     (w010 <= min_weight) || (w011 <= min_weight) ||
     (w100 <= min_weight) || (w101 <= min_weight) ||
     (w110 <= min_weight) || (w111 <= min_weight)) {
    return false;
  }

  float u, v, w;
  u = (gx - gx0);
  v = (gy - gy0);
  w = (gz - gz0);

  tsdf = (1 - u) * (1 - v) * (1 - w) * f000 +
         (1 - u) * (1 - v) * (    w) * f001 +
         (1 - u) * (    v) * (1 - w) * f010 +
         (1 - u) * (    v) * (    w) * f011 +
         (    u) * (1 - v) * (1 - w) * f100 +
         (    u) * (1 - v) * (    w) * f101 +
         (    u) * (    v) * (1 - w) * f110 +
         (    u) * (    v) * (    w) * f111;

  return true;
}

__device__ bool surface_normal(int volume_size, float volume_dimension,
                               float voxel_dimension, float min_weight,
                               const Voxel *volume, Vertex center,
                               float vx, float vy, float vz,
                               float &nx, float &ny, float &nz) {
  // Convert Vertex Position to Grid Position
  float gx, gy, gz;
  gx = (vx + (volume_dimension / 2.0f) - center.x) / voxel_dimension;
  gy = (vy + (volume_dimension / 2.0f) - center.y) / voxel_dimension;
  gz = (vz + (volume_dimension / 2.0f) - center.z) / voxel_dimension;

  float gx0, gy0, gz0;
  float gx1, gy1, gz1;

  gx0 = floor(gx);
  gx1 = gx0 + 1.0f;
  gy0 = floor(gy);
  gy1 = gy0 + 1.0f;
  gz0 = floor(gz);
  gz1 = gz0 + 1.0f;

  int i;
  float f000, f001, f010, f011, f100, f101, f110, f111;
  float w000, w001, w010, w011, w100, w101, w110, w111;

  i = (int)gx0 + (int)gy0 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f000 = UNCOMPRESS_VALUE(volume[i]);
    w000 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f000 = 0.0f;
    w000 = 0.0f;
  }

  i = (int)gx0 + (int)gy0 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f001 = UNCOMPRESS_VALUE(volume[i]);
    w001 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f001 = 0.0f;
    w001 = 0.0f;
  }

  i = (int)gx0 + (int)gy1 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f010 = UNCOMPRESS_VALUE(volume[i]);
    w010 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f010 = 0.0f;
    w010 = 0.0f;
  }

  i = (int)gx0 + (int)gy1 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f011 = UNCOMPRESS_VALUE(volume[i]);
    w011 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f011 = 0.0f;
    w011 = 0.0f;
  }

  i = (int)gx1 + (int)gy0 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f100 = UNCOMPRESS_VALUE(volume[i]);
    w100 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f100 = 0.0f;
    w100 = 0.0f;
  }

  i = (int)gx1 + (int)gy0 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f101 = UNCOMPRESS_VALUE(volume[i]);
    w101 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f101 = 0.0f;
    w101 = 0.0f;
  }

  i = (int)gx1 + (int)gy1 * volume_size +
      (int)gz0 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f110 = UNCOMPRESS_VALUE(volume[i]);
    w110 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f110 = 0.0f;
    w110 = 0.0f;
  }

  i = (int)gx1 + (int)gy1 * volume_size +
      (int)gz1 * volume_size * volume_size;
  if (i > 0 && i < volume_size * volume_size * volume_size) {
    f111 = UNCOMPRESS_VALUE(volume[i]);
    w111 = UNCOMPRESS_WEIGHT(volume[i]);
  }
  else {
    f111 = 0.0f;
    w111 = 0.0f;
  }

  if((w000 <= min_weight) || (w001 <= min_weight) ||
     (w010 <= min_weight) || (w011 <= min_weight) ||
     (w100 <= min_weight) || (w101 <= min_weight) ||
     (w110 <= min_weight) || (w111 <= min_weight)) {
    return false;
  }

  float u, v, w;
  u = (gx - gx0);
  v = (gy - gy0);
  w = (gz - gz0);

  nx = (1 - v) * (1 - w) * f000 +
       (1 - v) * (    w) * f001 +
       (    v) * (1 - w) * f010 +
       (    v) * (    w) * f011 +
      -(1 - v) * (1 - w) * f100 +
      -(1 - v) * (    w) * f101 +
      -(    v) * (1 - w) * f110 +
      -(    v) * (    w) * f111;

  ny = (1 - u) * (1 - w) * f000 +
       (1 - u) * (    w) * f001 +
      -(1 - u) * (1 - w) * f010 +
      -(1 - u) * (    w) * f011 +
       (    u) * (1 - w) * f100 +
       (    u) * (    w) * f101 +
      -(    u) * (1 - w) * f110 +
      -(    u) * (    w) * f111;

  nz = (1 - u) * (1 - v) * f000 +
      -(1 - u) * (1 - v) * f001 +
       (1 - u) * (    v) * f010 +
      -(1 - u) * (    v) * f011 +
       (    u) * (1 - v) * f100 +
      -(    u) * (1 - v) * f101 +
       (    u) * (    v) * f110 +
      -(    u) * (    v) * f111;

  return true;
}

__global__ void RayCaster(float max_distance, float max_truncation,
                          int volume_size, float volume_dimension,
                          float voxel_dimension, float min_weight,
                          int width, int height, float fx, float fy,
                          float cx, float cy, Vertex center,
                          const Voxel *volume, Vertices model_vertices,
                          Normals model_normals, Color *normal_map) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int x = tx + bx * BLOCK_WIDTH;
  int y = ty + by * BLOCK_WIDTH;
  int i = x + y * width;

  if ((x < width) && (y < height)) {
    // Initialize Normal Map
    Color normal_color;
    normal_color.r = normal_color.g = normal_color.b = 0;

    // Initialize Model Vertex and Normal
    Vertex model_vertex;
    Vector model_normal;

    model_vertex.x = model_vertex.y = model_vertex.z = 0.0f;
    model_normal.x = model_normal.y = model_normal.z = 0.0f;

    // Ray Starting Position
    Vertex ray_start;
    ray_start.x = T[0][3];
    ray_start.y = T[1][3];
    ray_start.z = T[2][3];

    // Ray Direction
    Vector ray_direction, ray_temp;
    ray_temp.x = (x - cx) / fx;
    ray_temp.y = (y - cy) / fy;
    ray_temp.z = 1.0f;

    ray_direction.x = T[0][0] * ray_temp.x +
                      T[0][1] * ray_temp.y +
                      T[0][2] * ray_temp.z;
    ray_direction.y = T[1][0] * ray_temp.x +
                      T[1][1] * ray_temp.y +
                      T[1][2] * ray_temp.z;
    ray_direction.z = T[2][0] * ray_temp.x +
                      T[2][1] * ray_temp.y +
                      T[2][2] * ray_temp.z;

    // Normalize Ray Direction
    float ray_inorm = rsqrt(ray_direction.x * ray_direction.x +
                            ray_direction.y * ray_direction.y +
                            ray_direction.z * ray_direction.z);

    ray_direction.x *= ray_inorm;
    ray_direction.y *= ray_inorm;
    ray_direction.z *= ray_inorm;

    // Ray Casting
    float time = 0.0f, time_last = 0.0f;
    float tsdf, tsdf_last = 0.0f;
    while (time < max_distance) {
      // Determine Vertex Position
      Vertex vertex_position;
      vertex_position.x = ray_start.x + ray_direction.x * time;
      vertex_position.y = ray_start.y + ray_direction.y * time;
      vertex_position.z = ray_start.z + ray_direction.z * time;

      // Determine Grid Position
      Vertex grid_position;
      grid_position.x = (vertex_position.x + (volume_dimension / 2.0f) -
                         center.x) / voxel_dimension;
      grid_position.y = (vertex_position.y + (volume_dimension / 2.0f) -
                         center.y) / voxel_dimension;
      grid_position.z = (vertex_position.z + (volume_dimension / 2.0f) -
                         center.z) / voxel_dimension;

      if ((grid_position.x) < 1 || (grid_position.x >= volume_size - 1)) {
        time_last = time;
        time += max_truncation;
        continue;
      }

      if ((grid_position.y) < 1 || (grid_position.y >= volume_size - 1)) {
        time_last = time;
        time += max_truncation;
        continue;
      }

      if ((grid_position.z) < 1 || (grid_position.z >= volume_size - 1)) {
        time_last = time;
        time += max_truncation;
        continue;
      }

      if (!interpolate(volume_size, volume_dimension, voxel_dimension,
                       min_weight, volume, center,
                       vertex_position.x, vertex_position.y, vertex_position.z,
                       tsdf)) {
        time_last = time;
        time += max_truncation;
        continue;
      }

      if (tsdf_last < 0.0f && tsdf >= 0.0f)
        break;

      if (tsdf_last > 0.0f && tsdf <= 0.0f) {
        // Determine Time
        float t = time_last -(((time - time_last) * tsdf_last) /
                              (tsdf - tsdf_last));

        // Determine Position
        model_vertex.x = ray_start.x + ray_direction.x * t;
        model_vertex.y = ray_start.y + ray_direction.y * t;
        model_vertex.z = ray_start.z + ray_direction.z * t;

        // Determine Normal
        if (surface_normal(volume_size, volume_dimension,
                           voxel_dimension, min_weight, volume, center,
                           model_vertex.x, model_vertex.y, model_vertex.z,
                           model_normal.x, model_normal.y, model_normal.z)) {
          // Normalize
          float inorm = rsqrt(model_normal.x * model_normal.x +
                              model_normal.y * model_normal.y +
                              model_normal.z * model_normal.z);

          if (isfinite(inorm)) {
            model_normal.x *= inorm;
            model_normal.y *= inorm;
            model_normal.z *= inorm;

            normal_color.r = ((model_normal.x + 1.0f) / 2.0f) * 255.0f;
            normal_color.g = ((model_normal.y + 1.0f) / 2.0f) * 255.0f;
            normal_color.b = ((model_normal.z + 1.0f) / 2.0f) * 255.0f;
          }
        }

        break;
      }

      tsdf_last = tsdf;
      time_last = time;

      if (ABS(tsdf) < 1.0f)
        time += voxel_dimension;
      else
        time += max_truncation;
    }

    model_vertices.x[i] = model_vertex.x;
    model_vertices.y[i] = model_vertex.y;
    model_vertices.z[i] = model_vertex.z;

    model_normals.x[i] = model_normal.x;
    model_normals.y[i] = model_normal.y;
    model_normals.z[i] = model_normal.z;

    if (normal_map != NULL)
      normal_map[i] = normal_color;
  }
}

void RayCastingKernel(float max_distance, float max_truncation,
                      int volume_size, float volume_dimension,
                      float voxel_dimension, float min_weight,
                      int width, int height, float fx, float fy,
                      float cx, float cy, Vertex center,
                      float *transformation, const Voxel *volume,
                      Vertices model_vertices, Normals model_normals,
                      Color *normal_map) {
  // Copy Transforms to Constant Memory
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(T, transformation, sizeof(float) * 16));

  // Launch Ray Casting Kernel
  int grid_width = (width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  RayCaster<<<grid_dim, block_dim>>>(max_distance, max_truncation, volume_size,
                                     volume_dimension, voxel_dimension,
                                     min_weight, width, height, fx, fy, cx, cy,
                                     center, volume, model_vertices,
                                     model_normals, normal_map);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
