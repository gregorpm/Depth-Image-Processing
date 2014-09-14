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
  min_depth = max_depth = depth[0];

  for (int i = 1; i < (width * height); i++) {
    min_depth = MIN(depth[i], min_depth);
    max_depth = MAX(depth[i], max_depth);
  }

  for (int i = 0; i < (width * height); i++)
    colorized_depth[i] = jet(depth[i], min_depth, max_depth);
}

void Colorize::Run(int width, int height, int min_depth, int max_depth,
                   const Depth *depth, Color *colorized_depth) {
  for (int i = 0; i < (width * height); i++)
    colorized_depth[i] = jet(depth[i], min_depth, max_depth);
}

Color Colorize::jet(float value, float min, float max) {
  Color color = { 0, 0, 0 };

 if ((value >= min) && (value <= max)) {
    float v = (value - min) / (max - min);

    if (v < 0.25f) {
      color.b = 255;
      color.g = (unsigned char)((4.0f * v) * 255.0f);
    } else if (v < 0.5f) {
      color.g = 255;
      color.b = (unsigned char)((1.0f + (4.0f * (0.25f - v))) * 255.0f);
    } else if (v < 0.75f) {
      color.g = 255;
      color.r = (unsigned char)((4.0f * (v - 0.5f)) * 255.0f);
    } else {
      color.r = 255;
      color.g = (unsigned char)((1.0f + (4.0f * (0.75f - v))) * 255.0f);
    }
  }

  return color;
}

} // namespace dip
