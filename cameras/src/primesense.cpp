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

#include <dip/cameras/primesense.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace openni;

namespace dip {

PrimeSense::PrimeSense(bool calibration) : enabled_(false) {
  // Initialize Variables
  for (int i = 0; i < SENSOR_TYPES; i++) {
    width_[i] = height_[i] = -1;
    fx_[i] = fy_[i] = -1.0f;
    running_[i] = false;
  }

  // Initialize OpenNI
  OpenNI::initialize();
  printf("Initializing Camera\n%s\n", OpenNI::getExtendedError());

  // Initialize Camera
  initialize(ANY_DEVICE, calibration);
}

PrimeSense::PrimeSense(int id, bool calibration) : enabled_(false) {
  // Initialize Variables
  for (int i = 0; i < SENSOR_TYPES; i++) {
    width_[i] = height_[i] = -1;
    fx_[i] = fy_[i] = -1.0f;
    running_[i] = false;
  }

  // Initialize OpenNI
  OpenNI::initialize();
  printf("Initializing Camera\n%s\n", OpenNI::getExtendedError());

  // Enumerate Devices
  Array<DeviceInfo> device_list;
  OpenNI::enumerateDevices(&device_list);

  if (device_list.getSize() < (id + 1)) {
    printf("Unable to Find Camera\n%s\n", OpenNI::getExtendedError());
    OpenNI::shutdown();
    return;
  }

  // Initialize Camera
  initialize(device_list[id].getUri(), calibration);
}

PrimeSense::PrimeSense(const char *uri, bool calibration) : enabled_(false) {
  // Initialize Variables
  for (int i = 0; i < SENSOR_TYPES; i++) {
    width_[i] = height_[i] = -1;
    fx_[i] = fy_[i] = -1.0f;
    running_[i] = false;
  }

  // Initialize OpenNI
  OpenNI::initialize();
  printf("Initializing Camera\n%s\n", OpenNI::getExtendedError());

  // Initialize Camera
  initialize(uri, calibration);
}

PrimeSense::~PrimeSense() {
  if (enabled_) {
    // Shutdown streams
    for (int i = 0; i < SENSOR_TYPES; i++) {
      if (stream_[i].isValid()) {
        if (running_[i])
          stream_[i].stop();
        stream_[i].destroy();
      }
    }

    OpenNI::shutdown();
  }
}

int PrimeSense::Update(Depth *depth) {
  if (enabled_ && running_[DEPTH_SENSOR]) {
    if (stream_[DEPTH_SENSOR].isValid()) {
      stream_[DEPTH_SENSOR].readFrame(&frame_);

      if (frame_.isValid()) {
        const DepthPixel *pixels =
          static_cast<const DepthPixel*>(frame_.getData());

        // Copy depth frame.
        memcpy(depth, pixels, sizeof(Depth) *
               width_[DEPTH_SENSOR] * height_[DEPTH_SENSOR]);

        return 0;
      }
    }
  }

  return -1;
}

int PrimeSense::Update(Color *color) {
  if (enabled_ && running_[COLOR_SENSOR]) {
    if (stream_[COLOR_SENSOR].isValid()) {
      stream_[COLOR_SENSOR].readFrame(&frame_);

      if (frame_.isValid()) {
        const RGB888Pixel* pixels =
          static_cast<const RGB888Pixel*>(frame_.getData());

        // Copy color frame.
        memcpy(color, pixels, sizeof(Color) *
               width_[COLOR_SENSOR] * height_[COLOR_SENSOR]);

        return 0;
      }
    }
  }

  return -1;
}

int PrimeSense::URI(char **uri) const {
  if (enabled_) {
    DeviceInfo device_info = device_.getDeviceInfo();

    int length = strlen(device_info.getUri());

    if (length > 0) {
      *uri = new char[length + 1];
      memcpy(*uri, device_info.getUri(), sizeof(char) * length);
      (*uri)[length] = '\0';

      return 0;
    }
  }

  return -1;
}

int PrimeSense::start(int sensor) {
  if (enabled_ && !running_[sensor]) {
    if (stream_[sensor].isValid()) {
      stream_[sensor].start();
      running_[sensor] = true;

      return 0;
    }
  }

  return -1;
}

int PrimeSense::stop(int sensor) {
  if (enabled_ && running_[sensor]) {
    if (stream_[sensor].isValid()) {
      stream_[sensor].stop();
      running_[sensor] = false;

      return 0;
    }
  }

  return -1;
}

int PrimeSense::resolution(int sensor, int width, int height) {
  // Stop sensor's stream.
  if (!stop(sensor)) {
    // Modify video mode's resolution.
    VideoMode video_mode = stream_[sensor].getVideoMode();
    video_mode.setResolution(width, height);
    stream_[sensor].setVideoMode(video_mode);

    // Restart sensor's stream.
    if (!start(sensor)) {
      // Update dimensions and focal lengths.
      VideoMode video_mode = stream_[sensor].getVideoMode();

      width_[sensor] = video_mode.getResolutionX();
      height_[sensor] = video_mode.getResolutionY();

      float horizontal_fov = stream_[sensor].getHorizontalFieldOfView();
      float vertical_fov = stream_[sensor].getVerticalFieldOfView();

      fx_[sensor] = width_[sensor] / (2.0f * tan(horizontal_fov / 2.0f));
      fy_[sensor] = height_[sensor] / (2.0f * tan(vertical_fov / 2.0f));

      return 0;
    }
  }

  return -1;
}

void PrimeSense::initialize(const char *uri, bool calibration) {
  // Open PrimeSense camera.
  if (device_.open(uri) == STATUS_OK) {
    enabled_ = true;

    // Create streams.
    SensorType sensor_types[SENSOR_TYPES];
    sensor_types[DEPTH_SENSOR] = (!calibration) ? SENSOR_DEPTH : SENSOR_IR;
    sensor_types[COLOR_SENSOR] = SENSOR_COLOR;

    for (int i = 0; i < SENSOR_TYPES; i++) {
      if (stream_[i].create(device_, sensor_types[i]) == STATUS_OK) {
        // Disable Mirroring
        stream_[i].setMirroringEnabled(false);

        // Start stream.
        stream_[i].start();
        running_[i] = true;

        // Grab dimensions of image.
        VideoMode video_mode = stream_[i].getVideoMode();

        width_[i] = video_mode.getResolutionX();
        height_[i] = video_mode.getResolutionY();

        // Compute Focal Lengths
        float horizontal_fov = stream_[i].getHorizontalFieldOfView();
        float vertical_fov = stream_[i].getVerticalFieldOfView();

        fx_[i] = width_[i] / (2.0f * tan(horizontal_fov / 2.0f));
        fy_[i] = height_[i] / (2.0f * tan(vertical_fov / 2.0f));
      }
      else {
        printf("Unable to Create Stream\n%s\n", OpenNI::getExtendedError());
      }
    }

    device_.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
  }
  else {
    printf("Unable to Open Camera\n%s\n", OpenNI::getExtendedError());
    OpenNI::shutdown();
  }
}

} // namespace dip
