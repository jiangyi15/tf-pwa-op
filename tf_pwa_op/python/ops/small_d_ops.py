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


def small_d(beta, j):
    w = small_d_weight(j)
    a, b = small_d_ops.small_d(beta, w, j)
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
    return ret

def delta_D(alpha, beta, gamma, j, la, lb, lc):
    w = small_d_weight(j)
    d = small_d(beta, j)
    la = [int(i*2+0.1) for i in la]
    lb = [int(i*2+0.1) for i in lb]
    lc = [int(i*2+0.1) for i in lc]
    x, y = small_d_ops.DeltaD(small_d=d, alpha=alpha, gamma=gamma, la=la, lb=lb, lc=lc, j=j)
    return tf.complex(x, y)
