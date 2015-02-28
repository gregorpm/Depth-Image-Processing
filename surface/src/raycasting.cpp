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

#include <dip/surface/raycasting.h>

using namespace Eigen;

namespace dip {

extern void RayCastingKernel(float max_distance, float max_truncation,
                             int volume_size, float volume_dimension,
                             float voxel_dimension, float min_weight,
                             int width, int height, float fx, float fy,
                             float cx, float cy, Vertex center,
                             float *transformation, const Voxel *volume,
                             Vertices model_vertices, Normals model_normals,
                             Color *normal_map);

void RayCasting::Run(float max_distance, float max_truncation, int volume_size,
                     float volume_dimension, float voxel_dimension,
                     float min_weight, int width, int height,
                     float fx, float fy, float cx, float cy, Vertex center,
                     const Matrix4f &transformation, const Voxel *volume,
                     Vertices model_vertices, Normals model_normals,
                     Color *normal_map) {
  float T[16];
  for (int m = 0; m < 4; m++) {
    for (int n = 0; n < 4; n++) {
      T[n + m * 4] = transformation(m, n);
    }
  }

  RayCastingKernel(max_distance, max_truncation, volume_size, volume_dimension,
                   voxel_dimension, min_weight, width, height, fx, fy, cx, cy,
                   center, T, volume, model_vertices, model_normals,
                   normal_map);
}

} // namespace dip
