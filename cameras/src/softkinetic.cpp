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

#include <dip/cameras/softkinetic.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mutex>
#include <thread>
#include <vector>

using namespace DepthSense;
using namespace std;

namespace dip {

const static int kFrameRate = 30;

static Context g_context;
static DepthNode g_depth_node;
static ColorNode g_color_node;
static IntrinsicParameters g_depth_parameters;
static IntrinsicParameters g_color_parameters;

static mutex g_depth_mutex, g_color_mutex;

static Depth *g_depth = NULL;
static Color *g_color = NULL;

volatile static bool g_running = false;
volatile static bool g_stop = false;
volatile static bool g_error = false;
volatile static bool g_updated[SENSOR_TYPES];

void InitializeCamera();
void StopCamera();
void OnNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data);
void OnNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data);

SoftKinetic::SoftKinetic() : enabled_(false) {
  if (!g_running) {
    thread(InitializeCamera).detach();
    while(!g_running && !g_error);

    if (g_running && !g_error) {
      enabled_ = true;

      width_[DEPTH_SENSOR] = g_depth_parameters.width;
      height_[DEPTH_SENSOR] = g_depth_parameters.height;
      fx_[DEPTH_SENSOR] = g_depth_parameters.fx;
      fy_[DEPTH_SENSOR] = g_depth_parameters.fy;

      width_[COLOR_SENSOR] = g_color_parameters.width;
      height_[COLOR_SENSOR] = g_color_parameters.height;
      fx_[COLOR_SENSOR] = g_color_parameters.fx;
      fy_[COLOR_SENSOR] = g_color_parameters.fy;
    }
  }
}

SoftKinetic::~SoftKinetic() {
  if (enabled_) {
    if (g_running) {
      g_stop = true;
      while(g_running);
    }
  }
}

int SoftKinetic::Update(Depth *depth) {
  if (enabled_) {
    if (g_depth_node.isSet()) {
      while(!g_updated[DEPTH_SENSOR]);
      if (g_updated[DEPTH_SENSOR]) {
        g_depth_mutex.lock();
        memcpy(depth, g_depth, sizeof(Depth) * width_[DEPTH_SENSOR] *
            height_[DEPTH_SENSOR]);
        g_updated[DEPTH_SENSOR] = false;
        g_depth_mutex.unlock();

        return 0;
      }
    }
  }

  return -1;
}

int SoftKinetic::Update(Color *color) {
  if (enabled_) {
    if (g_color_node.isSet()) {
      while(!g_updated[COLOR_SENSOR]);
      if (g_updated[COLOR_SENSOR]) {
        g_color_mutex.lock();
        memcpy(color, g_color, sizeof(Color) * width_[COLOR_SENSOR] *
            height_[COLOR_SENSOR]);
        g_updated[COLOR_SENSOR] = false;
        g_color_mutex.unlock();

        return 0;
      }
    }
  }

  return -1;
}

void InitializeCamera() {
  // Create instance of the DepthSense server on the localhost.
  g_context = Context::create();

  // Obtain a list of devices attached to the host.
  vector<Device> devices = g_context.getDevices();

  for (vector<Device>::const_iterator device_iter = devices.begin();
       device_iter != devices.end(); device_iter++) {
    Device device = *device_iter;

    StereoCameraParameters stereo_parameters =
        device.getStereoCameraParameters();

    // Obtain a list of the device's nodes.
    vector<Node> nodes = device.getNodes();

    for (vector<Node>::const_iterator node_iter = nodes.begin();
         node_iter != nodes.end(); node_iter++) {
      Node node = *node_iter;

      DepthNode depth_node = node.as<DepthNode>();
      if (depth_node.isSet() && !g_depth_node.isSet()) {
        g_depth_node = depth_node;
        g_depth_parameters = stereo_parameters.depthIntrinsics;
      }

      ColorNode color_node = node.as<ColorNode>();
      if (color_node.isSet() && !g_color_node.isSet()) {
        g_color_node = color_node;
        g_color_parameters = stereo_parameters.colorIntrinsics;
      }
    }
  }

  if (g_depth_node.isSet() && g_color_node.isSet()) {
    // Allocate frame buffers.
    g_depth = new Depth[g_depth_parameters.width * g_depth_parameters.height];
    g_color = new Color[g_color_parameters.width * g_color_parameters.height];
    g_updated[DEPTH_SENSOR] = g_updated[COLOR_SENSOR] = false;

    // Set the callbacks for newSampleReceived event.
    g_depth_node.newSampleReceivedEvent().connect(OnNewDepthSample);
    g_color_node.newSampleReceivedEvent().connect(OnNewColorSample);

    // Config nodes.
    g_depth_node.setEnableDepthMap(true);
    DepthNode::Configuration depth_config = g_depth_node.getConfiguration();

    depth_config.framerate = kFrameRate;
    depth_config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    depth_config.saturation = true;

    g_context.requestControl(g_depth_node);
    g_depth_node.setConfiguration(depth_config);
    g_context.releaseControl(g_depth_node);

    g_color_node.setEnableColorMap(true);
    ColorNode::Configuration color_config = g_color_node.getConfiguration();

    color_config.framerate = kFrameRate;
    color_config.compression = COMPRESSION_TYPE_MJPEG;
    color_config.powerLineFrequency = POWER_LINE_FREQUENCY_60HZ;

    g_context.requestControl(g_color_node);
    g_color_node.setConfiguration(color_config);
    g_context.releaseControl(g_color_node);

    // Add the nodes to the list of nodes that will be streamed.
    g_context.registerNode(g_depth_node);
    g_context.registerNode(g_color_node);

    g_context.startNodes();

    g_running = true;
    g_context.run();
  } else {
    g_error = true;
  }
}

void StopCamera() {
  g_context.stopNodes();

  if (g_depth_node.isSet())
    g_context.unregisterNode(g_depth_node);
  if (g_color_node.isSet())
    g_context.unregisterNode(g_color_node);

  if (g_depth != NULL) {
    delete [] g_depth;
    g_depth = NULL;
  }

  if (g_color != NULL) {
    delete [] g_color;
    g_color = NULL;
  }

  g_running = g_stop = g_error = false;
}

void OnNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data) {
  g_depth_mutex.lock();
  memcpy(g_depth, data.depthMap, sizeof(Depth) *
      g_depth_parameters.width * g_depth_parameters.height);

  int i = 0;
  for (int y = 0; y < g_depth_parameters.height; y++) {
    for (int x = 0; x < g_depth_parameters.width; x++, i++) {
      if (g_depth[i] >= 32001)
        g_depth[i] = 0;
    }
  }

  g_updated[DEPTH_SENSOR] = true;
  g_depth_mutex.unlock();

  if (g_stop)
    StopCamera();
}

void OnNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data) {
  g_color_mutex.lock();
  memcpy(g_color, data.colorMap, sizeof(Color) *
      g_color_parameters.width * g_color_parameters.height);

  int size = g_color_parameters.width * g_color_parameters.height;
  for (int i = 0; i < size; i++) {
    unsigned char tmp = g_color[i].r;
    g_color[i].r = g_color[i].b;
    g_color[i].b = tmp;
  }

  g_updated[COLOR_SENSOR] = true;
  g_color_mutex.unlock();

  if (g_stop)
    StopCamera();
}

} // namespace dip
