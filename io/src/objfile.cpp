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

#include <dip/io/objfile.h>

namespace dip {

OBJFile::OBJFile(const char* file_name, int mode) : enabled_(false),
                                                    file_(NULL) {
  // Open/Create OBJ File
  switch (mode) {
    case READ_OBJ:
      file_ = fopen(file_name, "r");
      break;
    case MODIFY_OBJ:
      file_ = fopen(file_name, "a+");
      break;
    case CREATE_OBJ:
      file_ = fopen(file_name, "w+");
      break;
  }

  if (file_ != NULL) {
    mode_ = mode;
    enabled_ = true;
  }
}

OBJFile::~OBJFile() {
  if (enabled_)
    fclose(file_);
}

int OBJFile::Read(Mesh *mesh) const {
  if (enabled_) {
    fseek(file_, 0, SEEK_SET);

    char type;

    Vertex vertex;
    Face face;

    while(true) {
      if(fscanf(file_, "%c ", &type) != 1)
        break;

      if(type == 'v') {
        // Read Vertex
        fscanf(file_, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);

        // Add to mesh
        mesh->AddVertex(vertex);
      }
      else if(type == 'f') {
        // Read Face
        fscanf(file_, "%d %d %d\n", &face.a, &face.b, &face.c);

        // Decrement vertex ids
        // With the OBJ format the first id
        // is one instead of zero.
        face.a--;
        face.b--;
        face.c--;

        // Add to mesh
        mesh->AddFace(face);
      }
      else {
        char junk;

        do {
          junk = fgetc(file_);
        } while((junk != '\n') && (junk != EOF));
      }
    }

    return 0;
  }

  return -1;
}

int OBJFile::Write(Mesh *mesh) const {
  if (enabled_ && (mode_ & (MODIFY_OBJ | CREATE_OBJ))) {
    fseek(file_, 0, SEEK_END);

    for (int n = 0; n < mesh->VertexCount(); n++) {
      Vertex vertex = mesh->GetVertex(n);
      fprintf(file_, "v %f %f %f\n", vertex.x, vertex.y, vertex.z);
    }

    for (int n = 0; n < mesh->FaceCount(); n++) {
      Face face = mesh->GetFace(n);
      fprintf(file_, "f %d %d %d\n", face.a + 1, face.b + 1, face.c + 1);
    }

    return 0;
  }

  return -1;
}

} // namespace dip
