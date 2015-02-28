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

// Interface for accessing depth/color cameras.

#ifndef DIP_CAMERAS_CAMERA_H
#define DIP_CAMERAS_CAMERA_H

#include <dip/common/types.h>

namespace dip {

enum SensorTypes {
  DEPTH_SENSOR,
  COLOR_SENSOR,
  SENSOR_TYPES,
};

class Camera {
public:
  virtual ~Camera() {};

  // Update depth image.
  //  depth - Buffer to hold the next depth image captured by the camera.
  //          The dimensions of the image should be the same as the dimensions
  //          returned by width() and height() functions.
  // Return zero when depth image is succesfully updated.
  virtual int Update(Depth *depth) = 0;

  // Update color image.
  //  color - Buffer to hold the next color image captured by the camera.
  //          The dimensions of the image should be the same as the dimensions
  //          returned by width() and height() functions.
  // Return zero when color image is succesfully updated.
  virtual int Update(Color *color) = 0;

  // Returns true if the camera was successfully enabled.
  virtual bool enabled() const = 0;

  // Dimensions of depth/color images.
  virtual int width(int sensor) const = 0;
  virtual int height(int sensor) const = 0;

  // Focal length of depth/color images.
  virtual float fx(int sensor) const = 0;
  virtual float fy(int sensor) const = 0;

  // Request image resolution.
  virtual int resolution(int sensor, int width, int height) = 0;
};

} // namespace dip

#endif // DIP_CAMERAS_CAMERA_H
