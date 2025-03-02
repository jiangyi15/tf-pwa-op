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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

#define STATUS_OK absl::OkStatus()

REGISTER_OP("SmallD")
    .Attr("T: {float, double}")
    .Input("beta: T")
    .Input("w: T")
    .Output("d: T")
    .Output("sincos: T")
    .Attr("j: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      int j;
      TF_RETURN_IF_ERROR(c->GetAttr("j", &j));
      auto shape = c->input(0);
      auto seq_length = c->Dim(shape, 0);
      auto output_shape1 = c->MakeShape({seq_length, j + 1, j + 1});
      auto output_shape2 = c->MakeShape({seq_length, j + 1});
      c->set_output(0, output_shape1);
      c->set_output(1, output_shape2);
      return STATUS_OK;
    });

REGISTER_OP("DeltaD")
    .Attr("T: {float, double}")
    .Input("small_d: T")
    .Input("alpha: T")
    .Input("gamma: T")
    .Input("la: int32")
    .Input("lb: int32")
    .Input("lc: int32")
    .Output("ret1: T")
    .Output("ret2: T")
    .Attr("j: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      int j;
      TF_RETURN_IF_ERROR(c->GetAttr("j", &j));
      auto shape = c->input(0);
      auto seq_length = c->Dim(shape, 0);
      auto na = c->Dim(c->input(3), 0);
      auto nb = c->Dim(c->input(4), 0);
      auto nc = c->Dim(c->input(5), 0);
      auto output_shape1 = c->MakeShape({seq_length, na, nb, nc});
      c->set_output(0, output_shape1);
      c->set_output(1, output_shape1);
      return STATUS_OK;
    });

REGISTER_OP("MonmentLambda")
    .Attr("T: {float, double}")
    .Input("m0: T")
    .Input("m1: T")
    .Input("m2: T")
    .Output("ret1: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return STATUS_OK;
    });

REGISTER_OP("MonmentLambdaGradient")
    .Attr("T: {float, double}")
    .Input("m0: T")
    .Input("m1: T")
    .Input("m2: T")
    .Output("ret1: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return STATUS_OK;
    });

REGISTER_OP("BlattWeisskopfBarrierFactor")
    .Attr("T: {float, double}")
    .Input("l: int32")
    .Input("q: T")
    .Input("q0: T")
    .Attr("d: float")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      auto seq_length = c->Dim(c->input(1), 0);
      auto nl = c->Dim(c->input(0), 0);
      c->set_output(0, c->MakeShape({seq_length, nl}));
      return STATUS_OK;
    });

// REGISTER_OP("BlattWeisskopfBarrierFactor_grad_q0")
//     .Attr("T: {float, double}")
//     .Input("l: int32")
//     .Input("q: T")
//     .Input("q0: T")
//     .Input("value: T")
//     .Input("grad: T")
//     .Attr("d: T")
//     .Output("out: T")
//     .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
//       c->set_output(0, c->input(1));
//       return Status::OK();
// });

// REGISTER_OP("BlattWeisskopfBarrierFactor_hess_q0")
//     .Attr("T: {float, double}")
//     .Input("l: int32")
//     .Input("q: T")
//     .Input("q0: T")
//     .Input("value: T")
//     .Input("grad: T")
//     .Input("value_grad: T")
//     .Input("grad_grad: T")
//     .Attr("d: T")
//     .Output("out: T")
//     .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
//       c->set_output(0, c->input(1));
//       return Status::OK();
// });
