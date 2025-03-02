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

template <typename T>
T blattweisskopf_f(const T z, const int l) {
  switch (l) {
    case 0: return 1;
    case 1: return sqrt(z + 1);
    case 2: return sqrt(z*(z+3)+9);
    case 3: return sqrt(z*(z*(z+6)+45)+225);
    case 4: return sqrt(z*(z*(z*(z+10)+135)+1575)+11025);
    case 5: return sqrt(z*(z*(z*(z*(z+15)+315)+6300)+99225)+893025);
    default: return 1.0;
  }
}

// CPU specialization of actual computation.
template <typename T> struct BlattWeisskopfFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &device, const int size,const int nl,const float d, const int *l, const T *q,
                  const T *q0, T *out) {
    for (int i = 0; i < size; ++i) {
      auto z = pow(q[i]*d,2);
      auto z0 = pow(q[i]*d,2);
      for (int li =0;li<nl;li++){
      out[i*nl +li] = blattweisskopf_f<T>(z0, l[li])/blattweisskopf_f<T>(z, l[li]);
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class BlattWeisskopfOp : public OpKernel {
public:
explicit BlattWeisskopfOp(OpKernelConstruction *context) : OpKernel(context) {

    OP_REQUIRES_OK(context, context->GetAttr("d", &d_));
  }

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &l = context->input(0); // (m,)
    const Tensor &q = context->input(1);     // (n,)
    const Tensor &q0 = context->input(2);     // (n,)

    int nl = l.dim_size(0);

    // Create an output tensor
    Tensor *output_tensor = NULL;
    auto shape1 = q.shape();
    shape1.AddDim(nl);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape1, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, q.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    BlattWeisskopfFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(q.NumElements()),
        nl, d_,
        l.flat<int>().data(), q.flat<T>().data(),
        q0.flat<T>().data(), output_tensor->flat<T>().data());
  }
  float d_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BlattWeisskopfBarrierFactor").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      BlattWeisskopfOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  extern template struct BlattWeisskopfFunctor<GPUDevice, T>;                          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BlattWeisskopfBarrierFactor").Device(DEVICE_GPU).TypeConstraint<T>("T"),                \
      BlattWeisskopfOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU
#endif // GOOGLE_CUDA

} // namespace functor
} // namespace tensorflow
