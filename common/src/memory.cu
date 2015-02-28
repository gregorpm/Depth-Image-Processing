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

#include <dip/common/error.h>

namespace dip {

void Allocate(void **buffer, int bytes) {
  // Allocate buffer.
  CUDA_ERROR_CHECK(cudaMalloc(buffer, bytes))
}

void Deallocate(void *buffer) {
  // Deallocate buffer.
  CUDA_ERROR_CHECK(cudaFree(buffer));
}

void Clear(void *buffer, int bytes) {
  // Clear buffer.
  CUDA_ERROR_CHECK(cudaMemset(buffer, 0, bytes));
}

void Set(void *buffer, int value, int bytes) {
  // Set buffer.
  CUDA_ERROR_CHECK(cudaMemset(buffer, value, bytes));
}

void Upload(void *dst, const void *src, int bytes) {
  // Copy buffer from CPU to GPU.
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void Download(void *dst, const void *src, int bytes) {
  // Copy buffer from GPU to CPU.
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void Copy(void *dst, const void *src, int bytes) {
  // Copy buffer from GPU to GPU.
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

void UploadImage(void *dst, const void *src, int width, int height,
                 int dst_pitch, int src_pitch) {
  // Copy Image from CPU to GPU.
  CUDA_ERROR_CHECK(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                width, height, cudaMemcpyHostToDevice));
}

void DownloadImage(void *dst, const void *src, int width, int height,
                   int dst_pitch, int src_pitch) {
  // Copy Image from GPU to CPU.
  CUDA_ERROR_CHECK(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                width, height, cudaMemcpyDeviceToHost));
}

void CopyImage(void *dst, const void *src, int width, int height,
               int dst_pitch, int src_pitch) {
  // Copy Image from GPU to GPU.
  CUDA_ERROR_CHECK(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                width, height, cudaMemcpyDeviceToDevice));
}

} // namespace dip
