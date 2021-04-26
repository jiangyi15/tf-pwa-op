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
template <typename T> struct SmallDFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int size, int j, const T *in, const T *w,
                  T *sincos, T *out) {
    auto n = (j + 1);
    for (int i = 0; i < size; ++i) {
      auto sa = sin(in[i] / 2);
      auto ca = cos(in[i] / 2);
      for (int l = 0; l < n; l++) {
        sincos[i * n + l] = pow(sa, l) * pow(ca, j - l);
      }
    }
    for (int i = 0; i < size; ++i) {
      for (int j1 = 0; j1 < n; j1++) {
        for (int j2 = 0; j2 < n; j2++) {
          out[i * n * n + j1 * n + j2] = 0.0;
          for (int l = 0; l < j + 1; l++) {
            out[i * n * n + j1 * n + j2] +=
                w[l * n * n + j1 * n + j2] * sincos[i * n + l];
          }
        }
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class SmallDOp : public OpKernel {
public:
  explicit SmallDOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("j", &j_));
  }

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &input_tensor = context->input(0); // (n,)
    const Tensor &w_tensor = context->input(1);     // (m1, m2, l)

    // Create an output tensor
    Tensor *output_tensor = NULL;
    Tensor *sincos_tensor = NULL;
    auto shape1 = input_tensor.shape();
    shape1.AddDim(j_ + 1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape1, &sincos_tensor));
    shape1.AddDim(j_ + 1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape1, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    SmallDFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()), j_,
        input_tensor.flat<T>().data(), w_tensor.flat<T>().data(),
        sincos_tensor->flat<T>().data(), output_tensor->flat<T>().data());
  }
  int j_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SmallD").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      SmallDOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  extern template struct SmallDFunctor<GPUDevice, T>;                          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SmallD").Device(DEVICE_GPU).TypeConstraint<T>("T"),                \
      SmallDOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif // GOOGLE_CUDA
} // namespace functor
} // namespace tensorflow
