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

#include <dip/segmentation/connectedcomponents.h>

#include <stdlib.h>

namespace dip {

void ConnectedComponents::Run(int max_difference, int width, int height,
                              const Depth *depth, std::vector<CC> &components,
                              int *labels) {
  // Determine Connected Components
  components.clear();

  // Initialize First Component
  labels[0] = 0;

  CC component;
  component.init(0, depth[0]);
  components.push_back(component);

  // Run Connected Component on First Row
  for (int x = 1; x < width; x++) {
    int i = x;

    // Initialize Label
    labels[i] = -1;

    // Check West
    if (DIFF(depth[i], depth[i - 1]) < max_difference) {
      // Current Pixel belongs to Neighbor's Component
      labels[i] = Find(components, labels[i - 1]);
    }

    // Initialize Component
    if (labels[i] == -1) {
      labels[i] = components.size();

      component.init(components.size(), depth[i]);
      components.push_back(component);
    }
    else {
      components.at(labels[i]).size++;
      components.at(labels[i]).mean += depth[i];
    }
  }

  // Run Connected Component on First Column
  for (int y = 1; y < height; y++) {
    int i = y * width;

    // Initialize Label
    labels[i] = -1;

    // Check North
    if (DIFF(depth[i], depth[i - width]) < max_difference) {
      // Current Pixel belongs to Neighbor's Component
      labels[i] = Find(components, labels[i - width]);
    }

    // Initialize Component
    if (labels[i] == -1) {
      labels[i] = components.size();

      component.init(components.size(), depth[i]);
      components.push_back(component);
    }
    else {
      components.at(labels[i]).size++;
      components.at(labels[i]).mean += depth[i];
    }
  }

  // Run Connected Component on the Rest of the Frame
  for (int y = 1; y < height; y++) {
    for (int x = 1; x < width; x++) {
      int i = x + y * width;

      // Initialize Label
      labels[i] = -1;

      // Check West
      if (DIFF(depth[i], depth[i - 1]) < max_difference) {
        // Current Pixel belongs to Neighbor's Component
        labels[i] = Find(components, labels[i - 1]);
      }

      // Check North
      if (DIFF(depth[i], depth[i - width]) < max_difference) {
        if (labels[i] != -1) {
          if (labels[i] != Find(components, labels[i - width]))
            labels[i] = Merge(components, labels[i], labels[i - width]);
        }
        else {
          labels[i] = Find(components, labels[i - width]);
        }
      }

      // Initialize Component
      if (labels[i] == -1) {
        labels[i] = components.size();

        component.init(components.size(), depth[i]);
        components.push_back(component);

      }
      else {
        components.at(labels[i]).size++;
        components.at(labels[i]).mean += depth[i];
      }
    }
  }

  // Update Components
  for (unsigned int n = 0; n < components.size(); n++) {
    if (components.at(n).root)
      components.at(n).mean /= components.at(n).size;
    else
      components.at(n).parent = Find(components, components.at(n).parent);
  }

  // Update Labels
  for (int n = 0; n < (width * height); n++)
    labels[n] = components.at(labels[n]).parent;
}

int ConnectedComponents::Find(std::vector<CC> &components, int a) {
  while (components.at(a).parent != a)
    a = components.at(a).parent;

  return a;
}

int ConnectedComponents::Merge(std::vector<CC> &components, int a, int b) {
  a = Find(components, a);
  b = Find(components, b);

  if (a < b) {
    components.at(b).parent = a;
    components.at(b).root = false;
    components.at(a).size += components.at(b).size;
    components.at(a).mean += components.at(b).mean;

    return a;
  }
  else {
    components.at(a).parent = b;
    components.at(a).root = false;
    components.at(b).size += components.at(a).size;
    components.at(b).mean += components.at(a).mean;

    return b;
  }
}

} // namespace dip
