// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "Object.h"

namespace owl {

  std::atomic<uint64_t> Object::nextAvailableID;

  Object::Object()
    : uniqueID(nextAvailableID++)
  {}

  size_t sizeOf(OWLDataType type)
  {
    if ((size_t)type >= (size_t)OWL_USER_TYPE_BEGIN)
      return (size_t)type - (size_t)OWL_USER_TYPE_BEGIN;
        
    switch(type) {
      
    case OWL_INT:
      return sizeof(int32_t);
    case OWL_INT2:
      return 2*sizeof(int32_t);
    case OWL_INT3:
      return 3*sizeof(int32_t);
    case OWL_INT4:
      return 4*sizeof(int32_t);
      
    case OWL_UINT:
      return sizeof(uint32_t);
    case OWL_UINT2:
      return 2*sizeof(uint32_t);
    case OWL_UINT3:
      return 3*sizeof(uint32_t);
    case OWL_UINT4:
      return 4*sizeof(uint32_t);
      
    case OWL_LONG:
      return sizeof(int64_t);
    case OWL_LONG2:
      return 2*sizeof(int64_t);
    case OWL_LONG3:
      return 3*sizeof(int64_t);
    case OWL_LONG4:
      return 4*sizeof(int64_t);
      
    case OWL_ULONG:
      return sizeof(uint64_t);
    case OWL_ULONG2:
      return 2*sizeof(uint64_t);
    case OWL_ULONG3:
      return 3*sizeof(uint64_t);
    case OWL_ULONG4:
      return 4*sizeof(uint64_t);
      
    case OWL_FLOAT:
      return sizeof(float);
    case OWL_FLOAT2:
      return 2*sizeof(float);
    case OWL_FLOAT3:
      return 3*sizeof(float);
    case OWL_FLOAT4:
      return 4*sizeof(float);
      
    case OWL_BUFFER:
      //      return sizeof();
      throw "device code for OWL_BUFFER type not yet implemented";
    case OWL_BUFFER_POINTER:
      return sizeof(void *);
    case OWL_GROUP:
      return sizeof(OptixTraversableHandle);
    case OWL_DEVICE:
      return sizeof(int32_t);
    default:
      throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                               +": not yet implemented for type #"
                               +std::to_string((int)type));
    }
  }

  std::string typeToString(OWLDataType type)
  {
    if (type >= OWL_USER_TYPE_BEGIN)
      return "(user defined type)";
    switch(type) {
      
    case OWL_INT:
      return "int";
    case OWL_INT2:
      return "int2";
    case OWL_INT3:
      return "int3";
    case OWL_INT4:
      return "int4";
      
    case OWL_UINT:
      return "uint";
    case OWL_UINT2:
      return "uint2";
    case OWL_UINT3:
      return "uint3";
    case OWL_UINT4:
      return "uint4";
      
    case OWL_LONG:
      return "long";
    case OWL_LONG2:
      return "long2";
    case OWL_LONG3:
      return "long3";
    case OWL_LONG4:
      return "long4";
      
    case OWL_ULONG:
      return "ulong";
    case OWL_ULONG2:
      return "ulong2";
    case OWL_ULONG3:
      return "ulong3";
    case OWL_ULONG4:
      return "ulong4";
      
    case OWL_FLOAT:
      return "float";
    case OWL_FLOAT2:
      return "float2";
    case OWL_FLOAT3:
      return "float3";
    case OWL_FLOAT4:
      return "float4";
      
    case OWL_BUFFER:
      return "OWLBuffer";
    case OWL_BUFFER_POINTER:
      return "OWLBufferPointer";
    case OWL_GROUP:
      return "OWLGroup";
    case OWL_DEVICE:
      return "OWLDevice";
    default:
      throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                               +": not yet implemented for type #"
                               +std::to_string((int)type));
    }
  }
  
} // ::owl


