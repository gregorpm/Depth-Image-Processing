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

#ifndef DIP_COMMON_MEMORY_H
#define DIP_COMMON_MEMORY_H

namespace dip {

extern void Allocate(void **buffer, int bytes);
extern void Deallocate(void *buffer);

extern void Clear(void *buffer, int bytes);
extern void Set(void *buffer, int value, int bytes);

extern void Upload(void *dst, const void *src, int bytes);
extern void Download(void *dst, const void *src, int bytes);
extern void Copy(void *dst, const void *src, int bytes);

extern void UploadImage(void *dst, const void *src, int width, int height,
                        int dst_pitch, int src_pitch);
extern void DownloadImage(void *dst, const void *src, int width, int height,
                          int dst_pitch, int src_pitch);
extern void CopyImage(void *dst, const void *src, int width, int height,
                      int dst_pitch, int src_pitch);

} // namespace dip

#endif // DIP_COMMON_MEMORY_H
