/*
Copyright (c) 2013-2014, Gregory P. Meyer
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

#include <dip/projects/facemodeling.h>

#include <stdio.h>

using namespace Eigen;

namespace dip {

FaceModeling::FaceModeling(int width, int height, float fx, float fy,
                           float cx, float cy) : width_(width), height_(height),
                           fx_(fx), fy_(fy), cx_(cx), cy_(cy),
                           initial_frame_(true) {
  // Allocate depth image on the CPU.
  depth_ = new Depth[width_ * height_];

  // Allocate depth images on the GPU.
  Allocate((void**)&segmented_depth_, sizeof(Depth) * width_ * height_);
  Allocate((void**)&filtered_depth_, sizeof(Depth) * width_ * height_);
  Allocate((void**)&denoised_depth_, sizeof(Depth) * width_ * height_);

  // Allocate model point-cloud on the GPU.
  Allocate((void**)&(vertices_.x), sizeof(float) * width_ * height_);
  Allocate((void**)&(vertices_.y), sizeof(float) * width_ * height_);
  Allocate((void**)&(vertices_.z), sizeof(float) * width_ * height_);

  Allocate((void**)&(normals_.x), sizeof(float) * width_ * height_);
  Allocate((void**)&(normals_.y), sizeof(float) * width_ * height_);
  Allocate((void**)&(normals_.z), sizeof(float) * width_ * height_);

  Allocate((void**)&(model_vertices_.x), sizeof(float) * width_ * height_);
  Allocate((void**)&(model_vertices_.y), sizeof(float) * width_ * height_);
  Allocate((void**)&(model_vertices_.z), sizeof(float) * width_ * height_);

  Allocate((void**)&(model_normals_.x), sizeof(float) * width_ * height_);
  Allocate((void**)&(model_normals_.y), sizeof(float) * width_ * height_);
  Allocate((void**)&(model_normals_.z), sizeof(float) * width_ * height_);

  // Allocate the volume on the GPU and
  // set the value of each voxel to zero.
  Allocate((void**)&volume_, sizeof(Voxel) *
           kVolumeSize * kVolumeSize * kVolumeSize);
  Clear((void*)volume_, sizeof(Voxel) *
        kVolumeSize * kVolumeSize * kVolumeSize);

  // Allocate normal map on GPU.
  Allocate((void**)&normal_map_, sizeof(Color) * width_ * height_);

  // Initialize rigid body transformation to the identity matrix.
  transformation_.setIdentity();
}

FaceModeling::~FaceModeling() {
  delete [] depth_;

  Deallocate((void*)(segmented_depth_));
  Deallocate((void*)(filtered_depth_));
  Deallocate((void*)(denoised_depth_));

  Deallocate((void*)vertices_.x);
  Deallocate((void*)vertices_.y);
  Deallocate((void*)vertices_.z);

  Deallocate((void*)normals_.x);
  Deallocate((void*)normals_.y);
  Deallocate((void*)normals_.z);

  Deallocate((void*)model_vertices_.x);
  Deallocate((void*)model_vertices_.y);
  Deallocate((void*)model_vertices_.z);

  Deallocate((void*)model_normals_.x);
  Deallocate((void*)model_normals_.y);
  Deallocate((void*)model_normals_.z);

  Deallocate((void*)volume_);

  Deallocate((void*)normal_map_);
}

void FaceModeling::Run(const Depth *depth, Color *normal_map) {
  // Segment the user's head from the depth image.
  if (head_segmentation_.Run(kMinDepth, kMaxDepth, kMaxDifference,
                             kMinHeadWidth, kMinHeadHeight,
                             kMaxHeadWidth, kMaxHeadHeight,
                             fx_, fy_, width_, height_, depth, depth_)) {
    printf("Unable to segment user's head from depth image\n");
    return;
  }

  // Upload the segmented depth image from the CPU to the GPU.
  Upload(segmented_depth_, depth_, sizeof(Depth) * width_ * height_);

  // Filter the depth image.
  variance_filter_.Run(width_, height_, segmented_depth_, filtered_depth_);
  bilateral_filter_.Run(kRegistrationBilateralFilterSigmaD,
                        kRegistrationBilateralFilterSigmaR,
                        width_, height_, filtered_depth_, denoised_depth_);

  // Construct the point-cloud by back-projecting the depth image.
  back_projection_.Run(width_, height_, fx_, fy_, cx_, cy_,
                       denoised_depth_, vertices_, normals_);

  // Compute the center of mass of the point-cloud.
  Vertex center = centroid_.Run(width_, height_, vertices_);

  // Register the current frame to the previous frame.
  if (!initial_frame_) {
    Matrix4f previous_transformation = transformation_;

    // Create Transformation that aligns the Current Frame
    // with the Previous Frame based on the centroids
    Matrix4f frame_transformation;
    frame_transformation.setIdentity();

    frame_transformation(0, 3) = previous_center_.x - center.x;
    frame_transformation(1, 3) = previous_center_.y - center.y;
    frame_transformation(2, 3) = previous_center_.z - center.z;

    // Approximate the current frame's global transformation.
    transformation_ = transformation_ * frame_transformation;

    // Perform ICP.
    if (icp_.Run(kICPIterations, kMinCorrespondences,
                 kMaxRotation, kMaxTranslation,
                 kMinErrorDifference,
                 kDistanceThreshold, kNormalThreshold,
                 fx_, fy_, cx_, cy_, width_, height_,
                 width_, height_, vertices_, normals_,
                 model_vertices_, model_normals_,
                 previous_transformation, transformation_)) {
      printf("Unable to integrate depth image\n");
      transformation_ = previous_transformation;
      return;
    }
  }
  else {
    // Set the center of the volume to the
    // center of mass of the initial frame.
    volume_center_ = center;
  }

  // Integrate the segmented depth image into the volumetric model.
  bilateral_filter_.Run(kIntegrationBilateralFilterSigmaD,
                        kIntegrationBilateralFilterSigmaR,
                        width_, height_, filtered_depth_, denoised_depth_);

  volumetric_.Run(kVolumeSize, kVolumeDimension, kVoxelDimension,
                  kMaxTruncation, kMaxWeight, width_,  height_,
                  fx_, fy_, cx_, cy_, volume_center_, transformation_.inverse(),
                  denoised_depth_, normals_, volume_);

  // Render the volume using ray casting. Update the model point-cloud
  // for the next frame's registration step. Generate the normal map of
  // the model to display the current state of the model to the user.
  ray_casting_.Run(kMaxDistance, kMaxTruncation, kVolumeSize, kVolumeDimension,
                   kVoxelDimension, width_, height_, fx_, fy_, cx_, cy_,
                   volume_center_, transformation_, volume_, model_vertices_,
                   model_normals_, normal_map_);

  // Download the normal map from the GPU to the CPU.
  Download(normal_map, normal_map_, sizeof(Color) * width_ * height_);

  // Update Model Center
  previous_center_ = center;

  initial_frame_ = false;
}

void FaceModeling::Model(Mesh *mesh) {
  // Allocate volume on the CPU.
  Voxel *volume = new Voxel[kVolumeSize * kVolumeSize * kVolumeSize];

  // Download the volume from the GPU to the CPU.
  Download((void*)volume, volume_, sizeof(Voxel) *
           kVolumeSize * kVolumeSize * kVolumeSize);

  // Construct mesh using marching cubes.
  marching_cubes_.Run(kVolumeSize, kVolumeDimension, kVoxelDimension,
                      volume_center_, volume, mesh);

  delete [] volume;
}

} // namespace dip
