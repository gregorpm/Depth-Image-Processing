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

#ifndef DIP_SURFACE_MARCHINGCUBES_H
#define DIP_SURFACE_MARCHINGCUBES_H

#include <dip/common/types.h>
#include <dip/common/macros.h>
#include <dip/surface/mesh.h>
#include <dip/surface/voxel.h>

namespace dip {

typedef struct {
  int ids[12];
} Cube;

class MarchingCubes {
public:
  MarchingCubes() {}
  ~MarchingCubes() {}

  void Run(int volume_size, float volume_dimension, float voxel_dimension,
           float min_weight, Vertex center, const Voxel *volume, Mesh *mesh);

private:
  Vertex Interpolate(const Voxel *volume, Vertex position_1, Vertex position_2,
                     int index_1, int index_2);
  bool Check(float min_weight, const Voxel *volume, const int *grid);
  int Neighbors(int id, int x, int y, int width, Cube *current, Cube *previous);

  DISALLOW_COPY_AND_ASSIGN(MarchingCubes);
};

} // namespace dip

#endif // DIP_SURFACE_MARCHINGCUBES_H
