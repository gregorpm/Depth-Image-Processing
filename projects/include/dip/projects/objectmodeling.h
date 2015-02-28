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

// This project generates a 3D model of an object by registering and integrating
// a sequence of depth images captured by a PrimeSense depth sensor. To
// construct a model the system performs the following steps:
//    1. Thresholding    - Depth pixels that are too close or too far from the
//                         camera are removed.
//    2. Filtering       - To reduce noise and improve registration, the depth
//                         image is filtered. A bilateral filter is used to
//                         remove noise while preserving depth discontinuities.
//    3. Depth Pyramid   - A coarse-to-fine registration is used to align the
//                         depth data. Each level of the pyramid is created by
//                         downsampling the previous level.
//    4. Back-Projection - To register the depth data, we need to create a
//                         point-cloud by back-projection the depth pixels to
//                         3D vertices.
//    5. Registration    - Iterative closest point (ICP) is used to determine
//                         the rigid body transformation that aligns the
//                         point-clouds. In order to maintain a real-time rate,
//                         a fast variant of ICP is used. This method assumes
//                         small angles of rotation between depth frames.
//    6. Integration     - Once the rigid body transformation is known, the
//                         depth images can be integrated into volumetric model.
//                         The volume contains millions of voxels, where each
//                         voxel contains a signed distance to the nearest
//                         surface computed along camera rays.
//    7. Rendering       - Ray casting is used to render the current state of
//                         the model. For each pixel, a corresponding ray is
//                         stepped through the volume. A surface is found when
//                         the ray steps from a positive signed distance to a
//                         negative signed distance.
// After constructing the volumetric model, a mesh is generated using marching
// cubes.

#ifndef DIP_PROJECTS_OBJECTMODELING_H
#define DIP_PROJECTS_OBJECTMODELING_H

#include <Eigen/Dense>

#include <dip/common/macros.h>
#include <dip/common/memory.h>
#include <dip/common/types.h>
#include <dip/filters/bilateral.h>
#include <dip/filters/threshold.h>
#include <dip/registration/icp.h>
#include <dip/sampling/downsample.h>
#include <dip/point_cloud/backprojection.h>
#include <dip/point_cloud/centroid.h>
#include <dip/surface/marchingcubes.h>
#include <dip/surface/mesh.h>
#include <dip/surface/raycasting.h>
#include <dip/surface/volumetric.h>
#include <dip/surface/voxel.h>

namespace dip {

// The threshold filter will remove depth pixels with a value smaller than
// kMinDepth or larger than kMaxDepth.
const int kMinDepth = 64;
const int kMaxDepth = 4096;

// The following parameters control the bilateral filter. The result of a
// bilateral filter is a weighted average of neighboring pixels. The weight of a
// neighboring pixel is based on its distance from the center pixel (controlled
// by kBilateralFilterSigmaD), and its difference from the center pixel
// (controlled by kBilateralFilterSigmaR). If kBilateralFilterSigmaD is small
// then distance pixels will have a larger weight. If kBilateralFilterSigmaR is
// small pixels with different depth values will have a larger weight.
const float kBilateralFilterSigmaD = 1.0f / (1.0f * 1.0f);
const float kBilateralFilterSigmaR = 1.0f / (50.0f * 50.0f);

// Number of levels in the depth pyramid.
const int kPyramidLevels = 3;

// The following parameters define how the images are downsampled. The value of
// kDownsampleFactor determines how much the image dimensions are reduced. The
// width and height of the image are divided by 2^kDownsampleFactor for each
// level of the pyramid. When the images are downsampled, neighboring pixels are
// averaged together. To avoid averaging over depth discontinuities, a
// neighboring pixel's value must be within kDownsampleMaxDifference of the
// center pixel to be include in the average.
const int kDownsampleFactor = 1;
const int kDownsampleMaxDifference = 100;

// ICP is controlled using the following parameters. kICPIterations determines
// the maximum number of iterations of ICP for each level of the pyramid. In
// order to fail gracefully, kMinCorrespondences, kMaxRotation, and
// kMaxTranslation are used to detect when ICP may produce poor results. If the
// number of point correspondences is below kMinCorrespondences, we assume ICP
// will not compute the correct rigid body transformation. We expect small
// amounts of motion between frames, so if the angles of rotation are larger
// than kMaxRotation or the translation is larger than kMaxTranslation, we
// assume ICP has failed to calculate an accurate rigid body transformation. To
// speedup the registration step, we stop iterating when the error stops
// decreasing rapidly. The value of kMinErrorDifference is used to determine
// when the error stops decreasing quickly. Point correspondences should be near
// each other in space, and their normals should point in a similar direction.
// kDistanceThreshold and kNormalThreshold are used to throw away bad
// corresponding points.
const int kICPIterations[kPyramidLevels] = { 15, 10, 5 };
const int kMinCorrespondences[kPyramidLevels + 1] = { 500, 250, 50, 10 };
const float kDistanceThreshold[kPyramidLevels + 1] = { 50.0f, 100.0f, 150.0f, 200.0f };
const float kNormalThreshold[kPyramidLevels + 1] = { 0.524f, 0.611f, 0.698f, 0.785f };
const float kMaxTranslation = 500.0f;
const float kMaxRotation = 1.047f;

// kVolumeSize determines the number of voxels along each dimension of the
// volume, therefore, the total number of voxels is kVolumeSize^3. The physical
// length of each side of the volume is defined by kVolumeDimension (in
// millimeters). kVoxelDimension is the physical size of a voxel. Each voxel
// contains a signed distance to the surface and a weight. To prevent surfaces
// from interfering with each other, the signed distances are truncated. When
// the value is truncated is determined by kMaxTruncation. Non-truncated values
// are averaged together to reduce noise in the reconstructed surface. The max
// truncation should be roughly the maximum amount of noise that could be added
// to a depth measurement. Also, kMaxTruncation should be larger than the
// dimension of a voxel. The value kMaxWeight determines how quickly the volume
// is updated with changes in the surface. A small value for kMaxWeight will
// enable the volume to be updated quickly when the scene is modified, but also,
// can quickly add errors to the reconstructed surface.
const int kVolumeSize = 256;
const float kVolumeDimension = 1024.0f;
const float kVoxelDimension = kVolumeDimension / kVolumeSize;
const float kMaxTruncation = 12.0f;
const float kMaxWeight = 256.0f;

// The following parameters control the ray caster. The maximum distance a ray
// will travel is defined by kMaxDistance.
const float kMaxDistance = 4096.0f;

class ObjectModeling {
public:
  // Initialize 3D modeling.
  //  width & height - The dimensions of the depth images.
  //  fx & fy        - Focal lengths of the depth camera.
  //  cx & cy        - Optical center of the depth camera.
  ObjectModeling(int width, int height, float fx, float fy, float cx, float cy);
  ~ObjectModeling();

  // Add a depth image into the 3D model.
  //  depth      - The depth image to be integrated into the model. The
  //               dimensions of the image should be the same as the dimensions
  //               specificed in the constructor.
  //  normal_map - Normal map of the model rendered from the same viewpoint as
  //               the depth image. The dimensions should be the same as the
  //               depth image.
  //  transform  - Transformation matrix from the current frames coordinate
  //               system to the global coordinate system.
  // Returns zero on success.
  int Run(const Depth *depth, Color *normal_map = NULL,
          Eigen::Matrix4f *transform = NULL);

  // Generate Model's mesh using Marching Cubes.
  //  mesh - Pointer to mesh data structure. Marching cubes will add the model's
  //         vertices, faces, and edges to data structure.
  void Model(Mesh *mesh);

private:
  // Modules used to construct the 3D model.
  Threshold threshold_filter_;
  Bilateral bilateral_filter_;
  Downsample downsample_;
  BackProjection back_projection_;
  Centroid centroid_;
  ICP icp_;
  Volumetric volumetric_;
  RayCasting ray_casting_;
  MarchingCubes marching_cubes_;

  // Depth images corresponding to the current depth frame.
  // These buffers are allocated on the GPU.
  Depth *depth_;
  Depth *denoised_depth_;
  Depth *depth_pyramid_[kPyramidLevels];

  // Point-clouds corresponding to the current depth image.
  // These buffers are allocated on the GPU. They are updated
  // in the back-projection step and used in the registration step.
  Vertices vertices_[kPyramidLevels];
  Normals normals_[kPyramidLevels];

  // Point-cloud corresponding to the current state of the model.
  // These bufferes are allocated on the GPU. They are updated
  // in the ray casting step and used in the registration step.
  Vertices model_vertices_;
  Normals model_normals_;

  // Volumetric representation of the model. Allocated on the GPU.
  // Update in the integration step, and rendered in the ray casting step.
  Voxel *volume_;

  // Normal map corresponding to the current state of the model,
  // and rendered from the current camera's coordinate system.
  // This buffer is allocated on the GPU, and it is updated in
  // the ray casting step.
  Color *normal_map_;

  // The physical position of the volume center.
  Vertex volume_center_;

  // The rigid body transformation computed by the registration step.
  Eigen::Matrix4f transformation_;

  // The dimensions of the depth image.
  int width_, height_;
  // The focal length and optical center of the depth camera.
  float fx_, fy_, cx_, cy_;

  // Flag used to determine whether or not the current frame
  // is the initial frame.
  bool initial_frame_;

  DISALLOW_COPY_AND_ASSIGN(ObjectModeling);
};

} // namespace dip

#endif // DIP_PROJECTS_OBJECTMODELING_H
