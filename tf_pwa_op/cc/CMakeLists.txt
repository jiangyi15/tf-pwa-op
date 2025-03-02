#
# Copyright (c) 2019, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)

set(customop_files
	kernels/small_d_kernels.cc
	kernels/small_d_kernels.cu
	kernels/Delta_D_kernels.cc
	kernels/Delta_D_kernels.cu
	kernels/monment_lambda_kernels.cc
	kernels/monment_lambda_kernels.cu
  kernels/blatt_weisskopf_barrier_factor_kernels.cc
  kernels/blatt_weisskopf_barrier_factor_kernels.cu
    ops/small_d_ops.cc
)

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DNDEBUG)

add_library(pwa_op SHARED
  ${customop_files}
)
set_target_properties(pwa_op PROPERTIES
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(pwa_op PRIVATE
  # -lcublas
  # -lcudart
  -l:libtensorflow_framework.so.2
  ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(
  pwa_op PRIVATE
  ${TF_PATH}/include
)

set_target_properties(pwa_op PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/ops)

set_target_properties(pwa_op PROPERTIES CUDA_ARCHITECTURES "70;72;75;80;86;90")
