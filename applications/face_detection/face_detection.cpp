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
#include <dip/cameras/camera.h>
#include <dip/cameras/dumpfile.h>
#include <dip/cameras/primesense.h>
#include <dip/common/types.h>
#include <dip/segmentation/facemasker.h>

using namespace cv;
using namespace dip;
using namespace std;

const int kWindowWidth = 640;
const int kWindowHeight = 480;

const bool kMasking = true;
const bool kDownsample = true;
const int kMinDepth = 256;
const int kMinPixels = 10;
const int kOpenSize = 2;
const int kHeadWidth = 150;
const int kHeadHeight = 150;
const int kHeadDepth = 100;
const int kFaceSize = 150;
const int kExtendedSize = 50;

const char kCascade[] = "haarcascade_frontalface_default.xml";

static void key_callback(GLFWwindow* window, int key, int scancode,
                         int action, int mods) {
  if ((key == GLFW_KEY_ESCAPE) && (action == GLFW_PRESS))
    glfwSetWindowShouldClose(window, GL_TRUE);
}

int main(int argc, char **argv) {
  if (argc > 2) {
    printf("Usage: %s [Dump File]\n", argv[0]);
    return -1;
  }

  // Initialize camera.
  Camera *camera = NULL;
  if (argc < 2)
    camera = new PrimeSense();
  else
    camera = new DumpFile(argv[1]);

  if (!camera->enabled()) {
    printf("Unable to Open Camera\n");
    return -1;
  }

  // Initialize buffers.
  Depth *depth = new Depth[camera->width(DEPTH_SENSOR) *
                           camera->height(DEPTH_SENSOR)];
  Color *color = new Color[camera->width(COLOR_SENSOR) *
                           camera->height(COLOR_SENSOR)];

  Depth *downsampled_depth = NULL;
  if (kDownsample) {
    downsampled_depth = new Depth[camera->width(DEPTH_SENSOR) *
                                  camera->height(DEPTH_SENSOR) / 4];
  }

  // Initialize face classifier.
  CascadeClassifier cascade;
  if (!cascade.load(kCascade)) {
    printf("Failed to load cascade classifier.\n");
    return -1;
  }

  // Initialize face masker.
  FaceMasker *masker = NULL;
  if (kMasking) {
    masker = new FaceMasker;
    Ptr<CascadeClassifier::MaskGenerator> masker_ptr(masker);
    cascade.setMaskGenerator(masker_ptr);
  }

  // Initialize GLFW
  if (!glfwInit()) {
    printf("Unable to Initialize GLFW.\n");
    return -1;
  }

  GLFWwindow *window = glfwCreateWindow(kWindowWidth, kWindowHeight,
                                        "Face Detection", NULL, NULL);

  if (!window) {
    printf("Unable to create window.\n");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback);

  // Initialize Texture
  GLuint texture;
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &texture);

  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
               camera->width(COLOR_SENSOR), camera->height(COLOR_SENSOR),
               0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

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

    if (kDownsample) {
      int i = 0;
      for (int y = 0; y < camera->height(DEPTH_SENSOR) / 2; y++) {
        for (int x = 0; x < camera->width(DEPTH_SENSOR) / 2; x++, i++) {
          int j = (x << 1) + (y << 1) * camera->width(DEPTH_SENSOR);
          downsampled_depth[i] = depth[j];
        }
      }
    }

    // Detect faces in color image.
    Mat image(camera->height(COLOR_SENSOR), camera->width(COLOR_SENSOR),
              CV_8UC3, color);

    // Eliminate sub-images using depth image.
    if (kMasking) {
      Size window_size = cascade.getOriginalWindowSize();

      if (kDownsample) {
        masker->Run(kMinDepth, kMinPixels, kOpenSize, kHeadWidth, kHeadHeight,
            kHeadDepth, kFaceSize, kExtendedSize, window_size.width,
            camera->width(DEPTH_SENSOR) / 2, camera->height(DEPTH_SENSOR) / 2,
            (camera->fx(DEPTH_SENSOR) + camera->fy(DEPTH_SENSOR)) /4.0f,
            downsampled_depth, color);
      } else {
        masker->Run(kMinDepth, kMinPixels, kOpenSize, kHeadWidth, kHeadHeight,
            kHeadDepth, kFaceSize, kExtendedSize, window_size.width,
            camera->width(DEPTH_SENSOR), camera->height(DEPTH_SENSOR),
            (camera->fx(DEPTH_SENSOR) + camera->fy(DEPTH_SENSOR)) / 2.0f,
            depth, color);
      }
    }

    vector<Rect> faces;
    cascade.detectMultiScale(image, faces);

    glfwMakeContextCurrent(window);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -10.0f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(1.0f, 1.0f, 1.0f);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    camera->width(COLOR_SENSOR), camera->height(COLOR_SENSOR),
                    GL_RGB, GL_UNSIGNED_BYTE, color);

    glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
      glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
      glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
      glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    // Draw face rectangles.
    glColor3f(1.0f, 0.0f, 0.0f);
    for (unsigned int i = 0; i < faces.size(); i++) {
      float left = (float)faces[i].x / camera->width(COLOR_SENSOR);
      float right = (float)(faces[i].x + faces[i].width) /
                    camera->width(COLOR_SENSOR);
      float top = 1.0f - ((float)faces[i].y /
                  camera->height(COLOR_SENSOR));
      float bottom = 1.0f - ((float)(faces[i].y + faces[i].height) /
                     camera->height(COLOR_SENSOR));

      glBegin(GL_LINE_LOOP);
        glVertex3f(left, top, 0.0f);
        glVertex3f(right, top, 0.0f);
        glVertex3f(right, bottom, 0.0f);
        glVertex3f(left, bottom, 0.0f);
      glEnd();
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  delete camera;
  delete [] depth;
  delete [] color;
  if (kDownsample) delete [] downsampled_depth;

  return 0;
}
