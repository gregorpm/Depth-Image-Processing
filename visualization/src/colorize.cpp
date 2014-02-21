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

#include <dip/visualization/colorize.h>

namespace dip {

void Colorize::Run(int width, int height, const Depth *depth,
                   Color *colorized_depth) {
  int min_depth, max_depth;
  bool set = false;

  for (int i = 0; i < (width * height); i++) {
    if(set) {
      min_depth = MIN(depth[i], min_depth);
      max_depth = MAX(depth[i], max_depth);
    }
    else {
      min_depth = depth[i];
      max_depth = depth[i];

      set = true;
    }
  }

  for (int i = 0; i < (width * height); i++) {
    if ((depth[i] > min_depth) && (depth[i] <= max_depth)) {
      float hue = static_cast<float>(depth[i] - min_depth) / max_depth;
      hue -= static_cast<int>(hue);

      Color color = hsl2rgb(hue, 0.5f, 1.0f);

      colorized_depth[i].r = color.r;
      colorized_depth[i].g = color.g;
      colorized_depth[i].b = color.b;
    }
    else {
      colorized_depth[i].r = 0;
      colorized_depth[i].g = 0;
      colorized_depth[i].b = 0;
    }
  }
}

void Colorize::Run(int width, int height, int min_depth, int max_depth,
                   const Depth *depth, Color *colorized_depth) {
  for (int i = 0; i < (width * height); i++) {
    if ((depth[i] > min_depth) && (depth[i] <= max_depth)) {
      float hue = static_cast<float>(depth[i] - min_depth) / max_depth;
      hue -= static_cast<int>(hue);

      Color color = hsl2rgb(hue, 0.5f, 1.0f);

      colorized_depth[i].r = color.r;
      colorized_depth[i].g = color.g;
      colorized_depth[i].b = color.b;
    }
    else {
      colorized_depth[i].r = 0;
      colorized_depth[i].g = 0;
      colorized_depth[i].b = 0;
    }
  }
}

Color Colorize::hsl2rgb(float hue, float lightness, float saturation) {
  float rgb[3] = { 0.0f, 0.0f, 0.0f };
  float clr[3] = { hue + 1.0f / 3.0f, hue, hue - 1.0f / 3.0f };
  float x, y;

  if (lightness <= 0.5f)
    x = lightness * (1.0f + saturation);
  else
    x = lightness + saturation - (lightness * saturation);

  y = 2.0f * lightness - x;

  for (int i = 0; i < 3; i++) {
    if (clr[i] < 0.0f)
      clr[i] += 1.0f;
    if (clr[i] > 1.0f)
      clr[i] -= 1.0f;

    if (6.0f * clr[i] < 1.0f)
      rgb[i] = y + (x - y) * clr[i] * 6.0f;
    else if (2.0f * clr[i] < 1.0f)
      rgb[i] = x;
    else if (3.0f * clr[i] < 2.0f)
      rgb[i] = (y + (x - y) * ((2.0f / 3.0f) - clr[i]) * 6.0f);
    else
      rgb[i] = y;
  }

  Color output;

  output.r = (unsigned char)(rgb[0] * 255.0f);
  output.g = (unsigned char)(rgb[1] * 255.0f);
  output.b = (unsigned char)(rgb[2] * 255.0f);

  return output;
}

} // namespace dip
