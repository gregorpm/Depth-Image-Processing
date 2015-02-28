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

// This class simplifies the task of reading, modifying, and creating OBJ
// geometry files.

#ifndef DIP_IO_OBJFILE_H
#define DIP_IO_OBJFILE_H

#include <stdio.h>

#include <dip/common/macros.h>
#include <dip/surface/mesh.h>

namespace dip {

// OBJ file access modes.
enum OBJ_MODES {
  READ_OBJ = 1,    // Read an existing OBJ file.
  MODIFY_OBJ = 2,  // Read/Write an existing OBJ file.
  CREATE_OBJ = 4,  // Creates a new OBJ file.
};

class OBJFile {
public:
  // Opens a OBJ file.
  //  file_name - Name of OBJ file.
  //  mode      - File access mode (READ_OBJ, MODIFY_OBJ, CREATE_OBJ).
  OBJFile(const char* file_name, int mode);
  ~OBJFile();

  // Reads a mesh from the OBJ file.
  //  mesh - Pointer to mesh data structure.
  // Returns zero when successful.
  int Read(Mesh *mesh) const;

  // Writes a mesh into the OBJ file.
  //  mesh - Pointer to mesh data structure.
  // Returns zero when successful.
  int Write(Mesh *mesh) const;

  // Returns true if the OBJ file was successfully opened.
  bool enabled() const { return enabled_; }

private:
  int mode_;
  bool enabled_;

  FILE *file_;

  DISALLOW_COPY_AND_ASSIGN(OBJFile);
};

} // namespace dip

#endif // DIP_IO_HDF5WRAPPER_H
