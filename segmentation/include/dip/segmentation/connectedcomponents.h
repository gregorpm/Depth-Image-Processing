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

// Identify connected components within a depth image. Two neighboring pixels
// are considered connected when the difference between their depth values is
// below a threshold. Connected component labeling uses a disjoint-set data
// structure and a union-find algorithm to efficiently determine connected
// components.

#ifndef DIP_SEGMENTATION_CONNECTEDCOMPONENTS_H
#define DIP_SEGMENTATION_CONNECTEDCOMPONENTS_H

#include <vector>

#include <dip/common/macros.h>
#include <dip/common/types.h>

namespace dip {

// Component structure (disjoint-set)
typedef struct {
  int parent;  // Parent label
  int size;    // Number of pixels in component
  int mean;    // Average depth of component
  bool root;   // Root component flag

  // Initialize Component with a single pixel.
  //  label - Unique set label.
  //  depth - Depth value of pixel.
  void init(int label, int depth) {
    parent = label;
    size = 1;
    mean = depth;
    root = true;
  }
} CC;

class ConnectedComponents {
public:
  ConnectedComponents() {}
  ~ConnectedComponents() {}

  // Performs connected component labeling.
  //  max_difference - Maximum difference between neighbor depth values to be
  //                   considered connected.
  //  width & height - Dimensions of depth image.
  //  depth          - Depth image.
  //  component      - Vector of all components (including non-root
  //                   components) within the depth image.
  //  labels         - Label map used to identify which connected component a
  //                   pixel belongs to. Label map should have the same
  //                   dimensions as the depth image.
  void Run(int max_difference, int width, int height, const Depth *depth,
           std::vector<CC> &components, int *labels);

private:
  // Finds parent component of disjoint-set.
  //  component - Vector of all components.
  //  a         - Label of child component.
  // Returns the label of the parent component. If the child component is
  // the parent, then the returned label equals a.
  int Find(std::vector<CC> &components, int a);

  // Merges two components.
  //  component - Vector of all components.
  //  a & b     - Labels of the two components to be merged.
  // Returns the label of the parent component after the merge.
  int Merge(std::vector<CC> &components, int a, int b);

  DISALLOW_COPY_AND_ASSIGN(ConnectedComponents);
};

} // namespace dip

#endif // DIP_SEGMENTATION_CONNECTEDCOMPONENTS_H
