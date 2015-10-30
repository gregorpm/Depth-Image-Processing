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

// Standard Libraries
#include <stdio.h>
#include <stdlib.h>

// OpenGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// DIP
#include <dip/cameras/primesense.h>
#include <dip/cameras/softkinetic.h>
#include <dip/common/types.h>
#include <dip/io/hdf5dumper.h>
#include <dip/io/hdf5wrapper.h>
#include <dip/visualization/colorize.h>

using namespace dip;

const int kWindowWidth = 640;
const int kWindowHeight = 480;

const int kMinDepth = 64;
const int kMaxDepth = 8192;

static void key_callback(GLFWwindow* window, int key, int scancode,
                         int action, int mods) {
  if ((key == GLFW_KEY_ESCAPE) && (action == GLFW_PRESS))
    glfwSetWindowShouldClose(window, GL_TRUE);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <Dump File> <Compression Level>\n", argv[0]);
    return -1;
  }

  // Initialize camera.
#ifndef SOFTKINETIC
  Camera *camera = new PrimeSense();
#else
  Camera *camera = new SoftKinetic();
#endif

  if (!camera->enabled()) {
    printf("Unable to Open Camera\n");
    return -1;
  }

  // Initialize dump file.
  HDF5Wrapper dump(argv[1], CREATE_HDF5);
  dump.Compression(atoi(argv[2]));

  // Initialize dumper.
  HDF5Dumper dumper(&dump);

  if (!dump.enabled()) {
    printf("Unable to Create Dump File\n");
    return -1;
  }

  // Output dimensions of depth and color images.
  int width, height;

  width = camera->width(DEPTH_SENSOR);
  height = camera->height(DEPTH_SENSOR);

  dump.Write("WIDTH", "/INFORMATION/DEPTH_SENSOR", &width, H5T_NATIVE_INT);
  dump.Write("HEIGHT", "/INFORMATION/DEPTH_SENSOR", &height, H5T_NATIVE_INT);

  width = camera->width(COLOR_SENSOR);
  height = camera->height(COLOR_SENSOR);

  dump.Write("WIDTH", "/INFORMATION/COLOR_SENSOR", &width, H5T_NATIVE_INT);
  dump.Write("HEIGHT", "/INFORMATION/COLOR_SENSOR", &height, H5T_NATIVE_INT);

  // Output focal length of depth and color images.
  float fx, fy;

  fx = camera->fx(DEPTH_SENSOR);
  fy = camera->fy(DEPTH_SENSOR);

  dump.Write("FX", "/INFORMATION/DEPTH_SENSOR", &fx, H5T_NATIVE_FLOAT);
  dump.Write("FY", "/INFORMATION/DEPTH_SENSOR", &fy, H5T_NATIVE_FLOAT);

  fx = camera->fx(COLOR_SENSOR);
  fy = camera->fy(COLOR_SENSOR);

  dump.Write("FX", "/INFORMATION/COLOR_SENSOR", &fx, H5T_NATIVE_FLOAT);
  dump.Write("FY", "/INFORMATION/COLOR_SENSOR", &fy, H5T_NATIVE_FLOAT);

  // Initialize Buffers
  Depth *depth = new Depth[camera->width(DEPTH_SENSOR) *
                           camera->height(DEPTH_SENSOR)];
  Color *colorized_depth = new Color[camera->width(DEPTH_SENSOR) *
                                     camera->height(DEPTH_SENSOR)];
  Color *color = new Color[camera->width(COLOR_SENSOR) *
                           camera->height(COLOR_SENSOR)];

  // Initialize GLFW
  if (!glfwInit()) {
    printf("Unable to Initialize GLFW.\n");
    return -1;
  }

  GLFWwindow *window = glfwCreateWindow(kWindowWidth * 2, kWindowHeight,
                                        "Camera Capture", NULL, NULL);

  if (!window) {
    printf("Unable to create window.\n");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback);

  // Initialize Texture
  GLuint textures[2];
  glEnable(GL_TEXTURE_2D);
  glGenTextures(2, textures);

  glBindTexture(GL_TEXTURE_2D, textures[0]);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
               camera->width(COLOR_SENSOR), camera->height(COLOR_SENSOR),
               0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

  glBindTexture(GL_TEXTURE_2D, textures[1]);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
               camera->width(DEPTH_SENSOR), camera->height(DEPTH_SENSOR),
               0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

  Colorize colorize;
  int frame = 0;
  while (!glfwWindowShouldClose(window)) {
    // Update depth image.
    if (camera->Update(depth)) {
      printf("Unable to update depth image.\n");
      break;
    }

    // Update color image.
    if (camera->Update(color)) {
      printf("Unable to update color image.\n");
      break;
    }

    // Colorize depth image.
    colorize.Run(camera->width(DEPTH_SENSOR), camera->height(DEPTH_SENSOR),
                 depth, colorized_depth);

    glfwMakeContextCurrent(window);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -10.0f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, kWindowWidth, kWindowHeight);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    camera->width(COLOR_SENSOR), camera->height(COLOR_SENSOR),
                    GL_RGB, GL_UNSIGNED_BYTE, color);

    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
      glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
      glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
      glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();

    glViewport(kWindowWidth, 0, kWindowWidth, kWindowHeight);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    camera->width(DEPTH_SENSOR), camera->height(DEPTH_SENSOR),
                    GL_RGB, GL_UNSIGNED_BYTE, colorized_depth);

    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
      glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
      glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
      glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();

    // Update Dump File
    char group[64];
    sprintf(group, "/FRAME%04d", frame);

    hsize_t dimensions[3];

    // Dump depth image.
    dimensions[0] = (hsize_t)camera->height(DEPTH_SENSOR);
    dimensions[1] = (hsize_t)camera->width(DEPTH_SENSOR);

    dumper.Write("DEPTH", group, depth, sizeof(Depth) *
                 camera->width(DEPTH_SENSOR) * camera->height(DEPTH_SENSOR),
                 dimensions, 2, H5T_NATIVE_SHORT);

    // Dump color image.
    dimensions[0] = (hsize_t)camera->height(COLOR_SENSOR);
    dimensions[1] = (hsize_t)camera->width(COLOR_SENSOR);
    dimensions[2] = 3;

    dumper.Write("COLOR", group, color, sizeof(Color) *
                 camera->width(COLOR_SENSOR) * camera->height(COLOR_SENSOR),
                 dimensions, 3, H5T_NATIVE_UCHAR);

    glfwSwapBuffers(window);
    glfwPollEvents();

    frame++;
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  delete camera;
  delete [] depth;
  delete [] color;
  delete [] colorized_depth;

  return 0;
}
