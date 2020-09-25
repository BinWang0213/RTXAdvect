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

#include "SBTObject.h"

namespace owl {

  SBTObjectType::SBTObjectType(Context *const context,
                               ObjectRegistry &registry,
                               size_t varStructSize,
                               const std::vector<OWLVarDecl> &varDecls)
    : RegisteredObject(context,registry),
      varStructSize(varStructSize),
      varDecls(varDecls)
  {
    for (auto &var : varDecls)
      assert(var.name != nullptr);
    /* TODO: at least in debug mode, do some 'duplicate variable
       name' and 'overlap of variables' checks etc */
  }

  int SBTObjectType::getVariableIdx(const std::string &varName)
  {
    for (int i=0;i<varDecls.size();i++) {
      assert(varDecls[i].name);
      if (!strcmp(varName.c_str(),varDecls[i].name))
        return i;
    }
    return -1;
  }

  bool SBTObjectType::hasVariable(const std::string &varName)
  {
    return getVariableIdx(varName) >= 0;
  }

  /*! create one instance each of a given type's variables */
  std::vector<Variable::SP> SBTObjectType::instantiateVariables()
  {
    std::vector<Variable::SP> variables(varDecls.size());
    for (size_t i=0;i<varDecls.size();i++) {
      variables[i] = Variable::createInstanceOf(&varDecls[i]);
      assert(variables[i]);
    }
    return variables;
  }

  /*! this function is arguably the heart of the NG layer: given an
    SBT Object's set of variables, create the SBT entry that writes
    the given variables' values into the specified format, prorperly
    translating per-device data (buffers, traversable) while doing
    so */
  void SBTObjectBase::writeVariables(uint8_t *sbtEntryBase,
                                     int deviceID) const
  {
    for (auto var : variables) {
      auto decl = var->varDecl;
      var->writeToSBT(sbtEntryBase + decl->offset,deviceID);
    }
  }
  
} // ::owl
