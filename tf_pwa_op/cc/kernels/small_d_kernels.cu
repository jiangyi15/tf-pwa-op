/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "small_d.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void SmallDCudaKernel(const int size, const int j, const T* beta, const T* w, T* sincos, T* out) {
  auto n = (j+1);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
        for (int l=0;l<n;l++){
            sincos[i*n + l] = pow(sin( beta[i]/2),l) *pow(cos(beta[i]/2), j-l);
        }
    }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
        for (int j1=0;j1<n;j1++){
        for (int j2=0;j2<n;j2++){
            out[i*n*n + j1*n + j2] = 0.0;
        for (int l=0;l<j+1;l++){
            out[i*n*n + j1*n + j2] += w[l*n*n+j1*n+j2] * sincos[i*n+l];
        }
        }
        }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct SmallDFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, int j, const T* in, const T* w,T* sincos, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    SmallDCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, j, in, w, sincos, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct SmallDFunctor<GPUDevice, float>;
template struct SmallDFunctor<GPUDevice, double>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
