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

template <typename T>
__device__ T blattweisskopf_f(const T z, const int l){
   switch (l) {
    case 0: return 1;
    case 1: return sqrt(z + 1);
    case 2: return sqrt(z*(z+3)+9);
    case 3: return sqrt(z*(z*(z+6)+45)+225);
    case 4: return sqrt(z*(z*(z*(z+1)+2)+3)+4);
    case 5: return sqrt(z*(z*(z*(z*(z+1)+2)+3)+4)+5);
    default: return 1.0;
  }
}

// Define the CUDA kernel.
template <typename T>
__global__ void BlattWeisskopfCudaKernel(const int size, const int nl, const float d, const int *l, const T *q,
                  const T *q0, T *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
      auto z = pow(q[i]*d,2);
      auto z0 = pow(q[i]*d,2);
      for (int li =0;li<nl;li++){
      out[i*nl +li] = blattweisskopf_f<T>(z0, l[li])/blattweisskopf_f<T>(z, l[li]);
      }
    }
}



// Define the GPU implementation that launches the CUDA kernel.

template <typename T>
struct BlattWeisskopfFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& device, const int size, const int nl, const float d, const int *l, const T *q,
                  const T *q0, T *out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    BlattWeisskopfCudaKernel<T>
        <<<block_count, thread_per_block, 0, device.stream()>>>(size, nl, d, l,  q, q0, out);
  }
};


// Explicitly instantiate functors for the types of OpKernels registered.
template struct BlattWeisskopfFunctor<GPUDevice, float>;
template struct BlattWeisskopfFunctor<GPUDevice, double>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
