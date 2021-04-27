
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#endif // GOOGLE_CUDA

#include "small_d.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T> struct DeltaDFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int size, int j, const T* small_d, 
        const T* alpha, 
        const T* gamma,
        const int* la,
        const int* lb,
        const int* lc,
        T* out_r,
        T* out_i,
        int na, int nb, int nc
        ) {
    auto n = (j + 1);
    for (int i = 0; i < size; ++i) {
      for (int j1 = 0; j1 < na; j1++) {
        for (int j2 = 0; j2 < nb; j2++) {
        for (int j3 = 0; j3 < nc; j3++) {
          out_r[i * na * nb*nc + j1 * nb*nc + j2*nc + j3] = 0.0;
          out_i[i * na * nb*nc + j1 * nb*nc + j2*nc + j3] = 0.0;
        }
      }
    }
  }
        }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class DeltaDOp : public OpKernel {
public:
  explicit DeltaDOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("j", &j_));
  }

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &small_d_tensor = context->input(0); // (n, l1, l2)
    const Tensor &alpha = context->input(1); // (n,)
    const Tensor &gamma = context->input(2); // (n,)
    const Tensor &la = context->input(3); // (m0,)
    const Tensor &lb = context->input(4); // (m1,)
    const Tensor &lc = context->input(5);     // (m2,)

    auto na = la.dim_size(0);
    auto nb = lb.dim_size(0);
    auto nc = lc.dim_size(0); 

    std::cout << na << nb << nc<< std::endl;

    // Create an output tensor
    Tensor *output_r_tensor = NULL;
    Tensor *output_i_tensor = NULL;
    auto shape1 = alpha.shape();
    shape1.AddDim(j_ + 1);
    // OP_REQUIRES_OK(context,
    //                context->allocate_temp(beta.dtype(), shape1, &sincos_tensor));
    // shape1.AddDim(j_ + 1);
    // OP_REQUIRES_OK(context,
    //                context->allocate_temp(beta.dtype(), shape1, &small_d_tensor));
    auto shape2 = alpha.shape();

    std::cout << na << nb << nc<< std::endl;
    shape2.AddDim(na);
    shape2.AddDim(nb);
    shape2.AddDim(nc);
 OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape2, &output_r_tensor));
 OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape2, &output_i_tensor));
                
    // Do the computation.
    OP_REQUIRES(context, alpha.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    DeltaDFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(alpha.NumElements()), j_,
        small_d_tensor.flat<T>().data(),
        alpha.flat<T>().data(), 
        gamma.flat<T>().data(),
        la.flat<int32>().data(),
        lb.flat<int32>().data(),
        lc.flat<int32>().data(),
        output_r_tensor->flat<T>().data(),
        output_i_tensor->flat<T>().data(),
        na, nb, nc
        );
  }
  int j_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DeltaD").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      DeltaDOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  extern template struct SmallDFunctor<GPUDevice, T>;                          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DeltaD").Device(DEVICE_GPU).TypeConstraint<T>("T"),                \
      DeltaDOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif // GOOGLE_CUDA
} // namespace functor
} // namespace tensorflow
