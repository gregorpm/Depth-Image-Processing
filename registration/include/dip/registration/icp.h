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

#ifndef DIP_REGISTRATION_ICP_H
#define DIP_REGISTRATION_ICP_H

#include <Eigen/Dense>

#include <dip/common/types.h>
#include <dip/common/macros.h>

namespace dip {

class ICP {
public:
  ICP();
  ~ICP();

  int Run(int max_iterations,
          int min_correspondences_begin, int min_correspondences_end,
          float distance_threshold_begin, float distance_threshold_end,
          float normal_threshold_begin, float normal_threshold_end,
          float max_rotation, float max_translation,
          float fx, float fy, float cx, float cy,
          int src_width, int src_height,
          int dst_width, int dst_height,
          Vertices src_vertices, Normals src_normals,
          Vertices dst_vertices, Normals dst_normals,
          const Eigen::Matrix4f &previous_transformation,
          Eigen::Matrix4f &transformation);

private:
  Eigen::Matrix4f ConstructTransform(Eigen::VectorXf &x);

  float *buffer_[29];
  int bytes_;

  DISALLOW_COPY_AND_ASSIGN(ICP);
};

} // namespace dip

#endif // DIP_REGISTRATION_ICP_H
