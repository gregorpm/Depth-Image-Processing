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

#include <dip/registration/icp.h>

#include <math.h>

#include <dip/common/memory.h>
#include <dip/common/reduction.h>

using namespace Eigen;

namespace dip {

extern void ICPKernel(float distance_threshold, float normal_threshold,
                      float fx, float fy, float cx, float cy,
                      float *frame_transformation,
                      float *global_transformation,
                      int src_width, int src_height,
                      int dst_width, int dst_height,
                      Vertices src_vertices, Normals src_normals,
                      Vertices dst_vertices, Normals dst_normals,
                      float *buffer[29]);

ICP::ICP() : bytes_(0) {
  for (int n = 0; n < 29; n++)
    buffer_[n] = NULL;
}

ICP::~ICP() {
  for (int n = 0; n < 29; n++) {
    if (buffer_[n] != NULL)
      Deallocate((void*)buffer_[n]);
  }
}

int ICP::Run(int max_iterations,
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
             Eigen::Matrix4f &transformation) {
  int required_bytes = sizeof(float) * src_width * src_height;

  if (bytes_ < required_bytes) {
    bytes_ = required_bytes;

    for (int n = 0; n < 29; n++) {
      if (buffer_[n] != NULL)
        Deallocate((void*)buffer_[n]);

      Allocate((void**)&(buffer_[n]), bytes_);
    }
  }

  Matrix4f previous_inverse_transformation = previous_transformation.inverse();

  float previous_error = -1.0f;
  for (int i = 0; i < max_iterations; i++) {
    float alpha = (float)i / (float)max_iterations;

    float dthreshold = (1.0f - alpha) * distance_threshold_begin +
                       alpha * distance_threshold_end;
    float nthreshold = (1.0f - alpha) * normal_threshold_begin +
                       alpha * normal_threshold_end;
    float cthreshold = (1.0f - alpha) * min_correspondences_begin +
                       alpha * min_correspondences_end;

    Matrix4f frame_transformation = previous_inverse_transformation *
                                    transformation;

    float Tf[16], Tg[16];
    for (int m = 0; m < 4; m++) {
      for (int n = 0; n < 4; n++) {
        Tf[n + m * 4] = frame_transformation(m, n);
        Tg[n + m * 4] = transformation(m, n);
      }
    }

    ICPKernel(dthreshold, nthreshold, fx, fy, cx, cy, Tf, Tg,
              src_width, src_height, dst_width, dst_height,
              src_vertices, src_normals, dst_vertices, dst_normals,
              buffer_);

    MatrixXf ata(6, 6);
    VectorXf atb(6), x(6), btb(1);
    float count;

    int k = 0;
    for (int m = 0; m < 6; m++) {
      for (int n = m; n < 6; n++) {
        ata(m, n) = Reduce(src_width * src_height, buffer_[k++]);
      }
    }


    for (int m = 0; m < 6; m++) {
      for (int n = 0; n < m; n++) {
        ata(m, n) = ata(n, m);
      }
    }

    for (int n = 0; n < 6; n++) {
      atb(n) = Reduce(src_width * src_height, buffer_[k++]);
    }

    btb(0) = Reduce(src_width * src_height, buffer_[k++]);

    count = Reduce(src_width * src_height, buffer_[k++]);

    if (count < cthreshold)
      return -1;

    // Solve Linear System of Equations
    x = ata.llt().solve(atb);

    // Check Rotation
    for(int n = 0; n < 3; n++) {
      if(x(n) > max_rotation)
        return -1;
    }

    // Check Translation
    for(int n = 3; n < 6; n++) {
      if(x(n) > max_translation)
        return -1;
    }

    // Compute Residual
    float residual = sqrt(((x.transpose() * ata * x) -
                           2.0f * (x.transpose() * atb) + btb)(0));
    float error = residual / count;

    if (previous_error != -1.0f) {
      if (error > previous_error)
        break;
    }

    previous_error = error;

    // Update Transformation
    transformation = ConstructTransform(x) * transformation;
  }

  return 0;
}

Matrix4f ICP::ConstructTransform(VectorXf &x) {
  Matrix4f transform;
  transform.setIdentity();

  transform(0, 0) = cos(x(2)) * cos(x(1));
  transform(0, 1) =-sin(x(2)) * cos(x(0)) + cos(x(2)) * sin(x(1)) * sin(x(0));
  transform(0, 2) = sin(x(2)) * sin(x(0)) + cos(x(2)) * sin(x(1)) * cos(x(0));

  transform(1, 0) = sin(x(2)) * cos(x(1));
  transform(1, 1) = cos(x(2)) * cos(x(0)) + sin(x(2)) * sin(x(1)) * sin(x(0));
  transform(1, 2) =-cos(x(2)) * sin(x(0)) + sin(x(2)) * sin(x(1)) * cos(x(0));

  transform(2, 0) =-sin(x(1));
  transform(2, 1) = cos(x(1)) * sin(x(0));
  transform(2, 2) = cos(x(1)) * cos(x(0));

  transform(0, 3) = x(3);
  transform(1, 3) = x(4);
  transform(2, 3) = x(5);

  return transform;
}

} // namespace dip
