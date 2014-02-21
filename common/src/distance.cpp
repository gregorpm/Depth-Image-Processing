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

#include <dip/common/distance.h>
#include <limits>

using namespace std;

namespace dip {

void Distance::Run(int width, int height, const bool *mask,
                   unsigned int *distance) {
  // Initialize distances.
  for (int i = 0; i < (width * height); i++) {
    if (mask[i])
      distance[i] = 0;
    else
      distance[i] = numeric_limits<unsigned int>::max();
  }

  // First pass.
  for (int y = 1; y < (height - 1); y++) {
    for (int x = 1; x < (width - 1); x++) {
      int i = x + y * width;

      if (!mask[i])
        distance[i] = FirstPass(i, width, distance);
    }
  }

  // Second pass
  for (int y = height - 2; y > 0; y--) {
    for (int x = width - 2; x > 0; x--) {
      int i = x + y * width;

      if (!mask[i])
        distance[i] = SecondPass(i, width, distance);
    }
  }
}

unsigned int Distance::FirstPass(int i, int width, unsigned int *distance) {
  unsigned int w = distance[i - 1] + 1;
  unsigned int n = distance[i - width] + 1;
  unsigned int nw = distance[i - width - 1] + 1;
  unsigned int ne = distance[i - width + 1] + 1;

  return MIN(distance[i], MIN(MIN(n, w), MIN(ne, nw)));
}
unsigned int Distance::SecondPass(int i, int width, unsigned int *distance) {
  unsigned int e = distance[i + 1] + 1;
  unsigned int s = distance[i + width] + 1;
  unsigned int se = distance[i + width + 1] + 1;
  unsigned int sw = distance[i + width - 1] + 1;

  return MIN(distance[i], MIN(MIN(s, e), MIN(se, sw)));
}

} // namespace dip
