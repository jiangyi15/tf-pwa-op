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
__global__ void MonmentLambdaCudaKernel(const int size, const T* m0, const T* m1, const T* m2, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
        out[i] = (m0[i] *m0[i] - (m1[i]-m2[i])*(m1[i]-m2[i]))*(m0[i] *m0[i] - (m1[i]+m2[i])*(m1[i]+m2[i]));
    }
}
template <typename T>
__global__ void  MonmentLambdaGradCudaKernel(const int size, const T* m0, const T* m1, const T* m2, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
        out[i] = 4 * m0[i] *(m0[i]*m0[i] - m1[i]*m1[i] -m2[i]*m2[i]);
    }
}


// Define the GPU implementation that launches the CUDA kernel.

template <typename T>
struct MonmentLambdaFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* m0, const T* m1,const T* m2, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    MonmentLambdaCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, m0, m1, m2, out);
  }
};

template <typename T>
struct  MonmentLambdaGradFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* m0, const T* m1,const T* m2, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
     MonmentLambdaGradCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, m0, m1, m2, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MonmentLambdaFunctor<GPUDevice, float>;
template struct MonmentLambdaFunctor<GPUDevice, double>;
template struct MonmentLambdaGradFunctor<GPUDevice, float>;
template struct MonmentLambdaGradFunctor<GPUDevice, double>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
