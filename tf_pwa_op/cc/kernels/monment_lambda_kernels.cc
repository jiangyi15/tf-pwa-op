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
template <typename T> struct MonmentLambdaFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int size, const T *m0, const T *m1,
                  const T *m2, T *out) {
    for (int i = 0; i < size; ++i) {
      out[i] = (m0[i] * m0[i] - (m1[i] - m2[i]) * (m1[i] - m2[i])) *
               (m0[i] * m0[i] - (m1[i] + m2[i]) * (m1[i] + m2[i]));
    }
  }
};

template <typename T> struct MonmentLambdaGradFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int size, const T *m0, const T *m1,
                  const T *m2, T *out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 4 * m0[i] * (m0[i] * m0[i] - m1[i] * m1[i] - m2[i] * m2[i]);
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class MonmentLambdaOp : public OpKernel {
public:
  explicit MonmentLambdaOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &m0 = context->input(0); // (n,)
    const Tensor &m1 = context->input(1); // (n,)
    const Tensor &m2 = context->input(2); // (n,)

    // Create an output tensor
    Tensor *output_tensor = NULL;
    auto shape1 = m0.shape();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape1, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, m0.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    MonmentLambdaFunctor<Device, T>()(
        context->eigen_device<Device>(), static_cast<int>(m0.NumElements()),
        m0.flat<T>().data(), m1.flat<T>().data(), m2.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

template <typename Device, typename T>
class MonmentLambdaGradOp : public OpKernel {
public:
  explicit MonmentLambdaGradOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &m0 = context->input(0); // (n,)
    const Tensor &m1 = context->input(1); // (n,)
    const Tensor &m2 = context->input(2); // (n,)

    // Create an output tensor
    Tensor *output_tensor = NULL;
    auto shape1 = m0.shape();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape1, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, m0.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    MonmentLambdaGradFunctor<Device, T>()(
        context->eigen_device<Device>(), static_cast<int>(m0.NumElements()),
        m0.flat<T>().data(), m1.flat<T>().data(), m2.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MonmentLambda").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
      MonmentLambdaOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  extern template struct MonmentLambdaFunctor<GPUDevice, T>;                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MonmentLambda").Device(DEVICE_GPU).TypeConstraint<T>("T"),         \
      MonmentLambdaOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU
#endif // GOOGLE_CUDA

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MonmentLambdaGradient").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MonmentLambdaGradOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  extern template struct MonmentLambdaGradFunctor<GPUDevice, T>;               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MonmentLambdaGradient").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MonmentLambdaGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU
#endif // GOOGLE_CUDA

} // namespace functor
} // namespace tensorflow
