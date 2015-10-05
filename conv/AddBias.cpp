// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"
#include "THClKernels.h"
#include "conv/AddBias.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL
#define PUBLIC

static std::string getKernelTemplate();

PUBLIC VIRTUAL AddBias::~AddBias() {
}
PUBLIC VIRTUAL void AddBias::forward(
    int batchSize, int numFilters, int outputHeight, int outputWidth,
    THClTensor *output,
    THClTensor *bias
      ) {
  StatefulTimer::timeCheck("AddBias::forward begin");

  THClKernels k(state, kernel);
  k.in(batchSize * numFilters * outputHeight * outputWidth);
  k.in(numFilters);
  k.in(outputHeight * outputWidth);
  k.inout(output);
  k.in(bias);
  int globalSize = batchSize * numFilters * outputHeight * outputWidth;
  int workgroupSize = 64;
  int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
  kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
//  cl->finish();

  StatefulTimer::timeCheck("AddBias::forward after repeatedAdd");
}
PUBLIC AddBias::AddBias(THClState *state, int device, ClConvolver *conv)  {
  this->state = state;
  this->device = device;
  this->conv = conv;

  string uniqueName = "AddBias.per_element_add";
  EasyCL *cl = THClState_getClv2(state, device);
  if(cl->kernelExists(uniqueName) ) {
    this->kernel = cl->getKernel(uniqueName);
    return;
  }
  TemplatedKernel kernelBuilder(cl);
  // kernelBuilder.set("", ""); // do this here...
  kernel = kernelBuilder.buildKernel(uniqueName, "per_element_add.cl",
    getKernelTemplate(), "repeated_add");
}

static std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel("kernel", "per_element_add.cl")
  // ]]]
  // generated using cog, from per_element_add.cl:
  const char * kernelSource =  
  "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
  "//\n" 
  "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
  "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
  "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
  "\n" 
  "kernel void per_element_add(const int N, global float *target, global const float *source) {\n" 
  "    const int globalId = get_global_id(0);\n" 
  "    if (globalId >= N) {\n" 
  "        return;\n" 
  "    }\n" 
  "    target[globalId] += source[globalId];\n" 
  "}\n" 
  "\n" 
  "// adds source to target\n" 
  "// tiles source as necessary, according to tilingSize\n" 
  "kernel void per_element_tiled_add(const int N, const int tilingSize, global float *target, global const float *source) {\n" 
  "    const int globalId = get_global_id(0);\n" 
  "    if (globalId >= N) {\n" 
  "        return;\n" 
  "    }\n" 
  "    target[globalId] += source[globalId % tilingSize];\n" 
  "}\n" 
  "\n" 
  "kernel void repeated_add(const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source) {\n" 
  "    const int globalId = get_global_id(0);\n" 
  "    if (globalId >= N) {\n" 
  "        return;\n" 
  "    }\n" 
  "    target[globalId] += source[ (globalId / repeatSize) % sourceSize ];\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]

  return kernelSource;
}

