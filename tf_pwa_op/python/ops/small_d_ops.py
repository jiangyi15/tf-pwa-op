# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use time_two ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


small_d_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('libpwa_op.so'))

import nvtx.plugins.tf as tf_nvtx

def small_d(beta, j):
    beta, d_id = tf_nvtx.ops.start(beta, "small_d_matrix")
    w = small_d_weight(j)
    a, b = small_d_ops.small_d(beta, w, j)
    a = tf_nvtx.ops.end(a, d_id)
    return a


@functools.lru_cache()
def small_d_weight(j):  # the prefactor in the d-function of β
    """
    For a certain j, the weight coefficient with index (:math:`m_1,m_2,l`) is
    :math:`w^{(j,m_1,m_2)}_{l} = (-1)^{m_1-m_2+k}\\frac{\\sqrt{(j+m_1)!(j-m_1)!(j+m_2)!(j-m_2)!}}{(j-m_1-k)!(j+m_2-k)!(m_1-m_2+k)!k!}`,
    and :math:`l` is an integer ranging from 0 to :math:`2j`.

    :param j: Integer :math:`2j` in the formula???
    :return: Of the shape (**j** +1, **j** +1, **j** +1). The indices correspond to (:math:`l,m_1,m_2`)
    """
    ret = np.zeros(shape=(j + 1, j + 1, j + 1))

    def f(x):
        return math.factorial(x >> 1)

    for m in range(-j, j + 1, 2):
        for n in range(-j, j + 1, 2):
            for k in range(max(0, n - m), min(j - m, j + n) + 1, 2):
                l = (2 * k + (m - n)) // 2
                sign = (-1) ** ((k + m - n) // 2)
                tmp = sign * math.sqrt(
                    1.0 * f(j + m) * f(j - m) * f(j + n) * f(j - n)
                )
                tmp /= f(j - m - k) * f(j + n - k) * f(k + m - n) * f(k)
                ret[l][(m + j) // 2][(n + j) // 2] = tmp
    return tf.convert_to_tensor(ret)

@functools.lru_cache()
def _cached_li(li):
    ret = [int(int(abs(i)*2+0.1) * np.sign(i)) for i in li]
    return tf.convert_to_tensor(ret)


def delta_D(alpha, beta, gamma, j, la, lb, lc):
    j = int(j*2+0.1)
    w = small_d_weight(j)
    d = small_d(beta, j)
    la = _cached_li(la)
    lb = _cached_li(lb)
    lc = _cached_li(lc)
    # print(j, la, lb, lc)
    x, y = small_d_ops.DeltaD(small_d=d, alpha=alpha, gamma=gamma, la=la, lb=lb, lc=lc, j=j)
    return tf.complex(x, y)


@tf.custom_gradient
def momentum_lambda(m0, m1, m2):

    """
.. math:
   \\lambda(a, b, c) = (a^2 - (b+c)^2)(a^2 - (b-c)^2)

.. math:
   \\frac{\\partial \\lambda(x, y,z )}{\\partial x } = 4 x (x^2 - y^2-  z^2)

    """
    def grad(g):
        return [ g * momentum_lambda_grad(m0, m1, m2),
                 g * momentum_lambda_grad(m1, m2, m0),
                 g * momentum_lambda_grad(m2, m0, m1)]
    x = small_d_ops.MonmentLambda(m0=m0, m1=m1, m2=m2)
    return x, grad

@tf.custom_gradient
def momentum_lambda_grad(m0, m1, m2):
    def grad(g):
        return [ g * (12*m0*m0-4*(m1*m1-m2*m2)),
                 g *(-8*m0*m1),
                 g * (-8*m0*m1)]
    x = small_d_ops.MonmentLambdaGradient(m0=m0, m1=m1, m2=m2)
    return x, grad

def get_relative_p2(m0, m1, m2):
    m0 = tf.convert_to_tensor(m0)
    m1 = tf.convert_to_tensor(m1)
    m2 = tf.convert_to_tensor(m2)
    shape = tf.broadcast_dynamic_shape(tf.broadcast_dynamic_shape(m0.shape, m1.shape), m2.shape)

    m0, m1, m2 = [ tf.broadcast_to(i, shape) for i in [m0,m1,m2]]
    if any([i.dtype == tf.float64 for i in [m0, m1, m2]]):
        m0, m1, m2 = [tf.cast(i, tf.float64) for i in [m0, m1, m2]]

    y = momentum_lambda(m0, m1, m2)
    return y/4/(m0*m0)

def get_relative_p(m0, m1, m2):
    return tf.sqrt(get_relative_p2(m0, m1, m2))

def blattweisskopf(l, q, q0, d=3.0):
    q = tf.convert_to_tensor(q)
    q0 = tf.convert_to_tensor(q0)
    shape = tf.broadcast_dynamic_shape(q.shape, q0.shape)

    q, q0 = [ tf.broadcast_to(i, shape) for i in [q, q0]]
    if any([i.dtype == tf.float64 for i in [q, q0]]):
        q, q0 = [tf.cast(i, tf.float64) for i in [q, q0]]

    return small_d_ops.BlattWeisskopfBarrierFactor(l=l, q=q, q0=q0, d=d)
