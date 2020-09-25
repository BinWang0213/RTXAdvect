# ======================================================================== #
# Copyright 2018-2020 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

# this helper script sets the following variables:
#
# OWL_INCLUDES - list of directories required to compile progs using owl
#
# OWL_LIBRARIES - list of libraries to link against when building owl programs

if (OWL_USE_TBB)
  include(configure_tbb)
endif()

set(OWL_INCLUDES
  # owl needs cuda:
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  # owl needs optix:
  ${OptiX_INCLUDE}
  # public API
  ${owl_dir}/owl/include
  # device API and common currently still include non-public header files
  ${owl_dir}/
  )
set(OWL_LIBRARIES
  owl_static
  )
if (TBB_FOUND)
  set(OWL_INCLUDES ${OWL_INCLUDES} ${TBB_INCLUDES})
  set(OWL_LIBRARIES ${OWL_LIBRARIES} ${TBB_LIBRARIES})
endif()

