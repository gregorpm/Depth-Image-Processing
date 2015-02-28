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

__constant__ float Tf[4][4];
__constant__ float Tg[4][4];

__constant__ float *buffer_ptr[29];

__global__ void Correspondence(float distance_threshold, float normal_threshold,
                               float fx, float fy, float cx, float cy,
                               int src_width, int src_height,
                               int dst_width, int dst_height,
                               Vertices src_vertices, Normals src_normals,
                               Vertices dst_vertices, Normals dst_normals) {
  // Get Block and Thread Id
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Calculate Row & Column
  int col = tx + bx * BLOCK_WIDTH;
  int row = ty + by * BLOCK_WIDTH;

  float A[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  float B = 0.0f, C = 0.0f;

  if ((col < src_width) && (row < src_height)) {
    int i = col + row * src_width;

    // Load vertex and normal
    Vertex src_vertex;
    src_vertex.x = src_vertices.x[i];
    src_vertex.y = src_vertices.y[i];
    src_vertex.z = src_vertices.z[i];

    Vector src_normal;
    src_normal.x = src_normals.x[i];
    src_normal.y = src_normals.y[i];
    src_normal.z = src_normals.z[i];

    if (src_vertex.z > 0) {
      // Convert from the source's coordinate system
      // to the the destination's coordinate system.
      Vertex src_vertex_f;

      src_vertex_f.x = Tf[0][0] * src_vertex.x +
                       Tf[0][1] * src_vertex.y +
                       Tf[0][2] * src_vertex.z +
                       Tf[0][3];
      src_vertex_f.y = Tf[1][0] * src_vertex.x +
                       Tf[1][1] * src_vertex.y +
                       Tf[1][2] * src_vertex.z +
                       Tf[1][3];
      src_vertex_f.z = Tf[2][0] * src_vertex.x +
                       Tf[2][1] * src_vertex.y +
                       Tf[2][2] * src_vertex.z +
                       Tf[2][3];

      // Project source vertex into the destination's image plane.
      int u = (int)roundf((fx * src_vertex_f.x + cx * src_vertex_f.z) /
                          src_vertex_f.z);
      int v = (int)roundf((fy * src_vertex_f.y + cy * src_vertex_f.z) /
                          src_vertex_f.z);

      // Check corresponding vertex
      if ((u >= 0 && u < dst_width) && (v >= 0 && v < dst_height)) {
        int j = u + v * dst_width;

        // Load corresponding vertex and normal
        Vertex dst_vertex_g;
        dst_vertex_g.x = dst_vertices.x[j];
        dst_vertex_g.y = dst_vertices.y[j];
        dst_vertex_g.z = dst_vertices.z[j];

        Vector dst_normal_g;
        dst_normal_g.x = dst_normals.x[j];
        dst_normal_g.y = dst_normals.y[j];
        dst_normal_g.z = dst_normals.z[j];

        if (dst_vertex_g.z > 0) {
          // Convert from the source's coordinate system
          // to the the global coordinate system.
          Vertex src_vertex_g;

          src_vertex_g.x = Tg[0][0] * src_vertex.x +
                           Tg[0][1] * src_vertex.y +
                           Tg[0][2] * src_vertex.z +
                           Tg[0][3];
          src_vertex_g.y = Tg[1][0] * src_vertex.x +
                           Tg[1][1] * src_vertex.y +
                           Tg[1][2] * src_vertex.z +
                           Tg[1][3];
          src_vertex_g.z = Tg[2][0] * src_vertex.x +
                           Tg[2][1] * src_vertex.y +
                           Tg[2][2] * src_vertex.z +
                           Tg[2][3];

          // Check Distance Threshold
          float distance = sqrt(pow((dst_vertex_g.x - src_vertex_g.x), 2) +
                                pow((dst_vertex_g.y - src_vertex_g.y), 2) +
                                pow((dst_vertex_g.z - src_vertex_g.z), 2));

          if (distance < distance_threshold) {
            // Check Normal Threshold
            Vector src_normal_g;

            src_normal_g.x = Tg[0][0] * src_normal.x +
                             Tg[0][1] * src_normal.y +
                             Tg[0][2] * src_normal.z;
            src_normal_g.y = Tg[1][0] * src_normal.x +
                             Tg[1][1] * src_normal.y +
                             Tg[1][2] * src_normal.z;
            src_normal_g.z = Tg[2][0] * src_normal.x +
                             Tg[2][1] * src_normal.y +
                             Tg[2][2] * src_normal.z;

            // Normalize
            float inorm = rsqrt(src_normal_g.x * src_normal_g.x +
                                src_normal_g.y * src_normal_g.y +
                                src_normal_g.z * src_normal_g.z);

            if (isfinite(inorm)) {
              src_normal_g.x *= inorm;
              src_normal_g.y *= inorm;
              src_normal_g.z *= inorm;

              float angle = dst_normal_g.x * src_normal_g.x +
                            dst_normal_g.y * src_normal_g.y +
                            dst_normal_g.z * src_normal_g.z;

              if (acos(angle) < normal_threshold) {
                // Compute a row of A
                A[0] = dst_normal_g.z * src_vertex_g.y -
                       dst_normal_g.y * src_vertex_g.z;
                A[1] = dst_normal_g.x * src_vertex_g.z -
                       dst_normal_g.z * src_vertex_g.x;
                A[2] = dst_normal_g.y * src_vertex_g.x -
                       dst_normal_g.x * src_vertex_g.y;
                A[3] = dst_normal_g.x;
                A[4] = dst_normal_g.y;
                A[5] = dst_normal_g.z;

                // Compute a row of B
                B = dst_normal_g.x * dst_vertex_g.x +
                    dst_normal_g.y * dst_vertex_g.y +
                    dst_normal_g.z * dst_vertex_g.z -
                    dst_normal_g.x * src_vertex_g.x -
                    dst_normal_g.y * src_vertex_g.y -
                    dst_normal_g.z * src_vertex_g.z;

                C = 1.0f;
              }
            }
          }
        }
      }
    }

    int k = 0;

    // Output ATA Matrix
    for (int m = 0; m < 6; m++) {
      for (int n = m; n < 6; n++) {
        buffer_ptr[k++][i] = A[m] * A[n];
      }
    }

    // Output ATB Matrix
    for(int n = 0; n < 6; n++)
      buffer_ptr[k++][i] = A[n] * B;

    // Output BTB Matrix
    buffer_ptr[k++][i] = B * B;

    // Output Correspondence Flag
    buffer_ptr[k++][i] = C;
  }
}

void ICPKernel(float distance_threshold, float normal_threshold,
               float fx, float fy, float cx, float cy,
               float *frame_transformation, float *global_transformation,
               int src_width, int src_height, int dst_width, int dst_height,
               Vertices src_vertices, Normals src_normals,
               Vertices dst_vertices, Normals dst_normals,
               float *buffer[29]) {
  // Copy Transforms to Constant Memory
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(Tf, frame_transformation,
                                      sizeof(float) * 16));
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(Tg, global_transformation,
                                      sizeof(float) * 16));

  // Copy buffer pointers to Constant Memory
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(buffer_ptr, buffer, sizeof(float*) * 29));

  // Launch ICP Kernel
  int grid_width = (src_width + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;
  int grid_height = (src_height + (BLOCK_WIDTH - 1)) / BLOCK_WIDTH;

  dim3 grid_dim(grid_width, grid_height, 1);
  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  Correspondence<<<grid_dim, block_dim>>>(distance_threshold, normal_threshold,
                                          fx, fy, cx, cy,
                                          src_width, src_height,
                                          dst_width, dst_height,
                                          src_vertices, src_normals,
                                          dst_vertices, dst_normals);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

} // namespace dip
