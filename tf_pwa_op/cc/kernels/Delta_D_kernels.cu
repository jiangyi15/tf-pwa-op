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
__global__ void DeltaDCudaKernel(const int size, const int j, const T* small_d,
        const T* alpha,
        const T* gamma,
        const int* la,
        const int* lb,
        const int* lc,
        T* out_r,
        T* out_i,
        const int na, const int nb, const int nc
        ) {
  auto n = (j+1);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
     for (int ja = 0; ja < na; ja++) {
        for (int jb = 0; jb < nb; jb++) {
        for (int jc = 0; jc < nc; jc++) {
          out_r[i * na * nb*nc + ja * nb*nc + jb*nc + jc] = 0.0;
          out_i[i * na * nb*nc + ja * nb*nc + jb*nc + jc] = 0.0;
          int ib = lb[jb];
          int ic = lc[jc];
          int delta = ib -ic;
          if (abs(delta) <= j) {
              int ia = la[ja];
              int idx = i*n*n + (ia+j)*n/2 + (delta+j)/2;
              T tmp = small_d[idx];
              T theta = 0.5*(ia * alpha[i] + delta * gamma[i]);
              out_r[i * na * nb*nc + ja * nb*nc + jb*nc + jc] = cos(theta) * tmp;
              out_i[i * na * nb*nc + ja * nb*nc + jb*nc + jc] = sin(theta) * tmp;
          }
        }
      }
    }
       }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct DeltaDFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, int j, const T* small_d,
        const T* alpha,
        const T* gamma,
        const int* la,
        const int* lb,
        const int* lc,
        T* out_r,
        T* out_i,
        int na, int nb, int nc
        )  {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    DeltaDCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, j, small_d, alpha, gamma, la, lb, lc, out_r, out_i, na, nb, nc);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct DeltaDFunctor<GPUDevice, float>;
template struct DeltaDFunctor<GPUDevice, double>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
