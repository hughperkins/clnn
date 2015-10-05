// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/Forward4.h"
//#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"
#include "conv/AddBias.h"
#include "conv/ClConvolver.h"

#include <sstream>
#include <string>
#include <iostream>
using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC
#define PUBLIC

static std::string getKernelTemplate();

PUBLIC VIRTUAL Forward4::~Forward4() {
  delete kernel;
  delete addBias;
}
static void printTensor(THClState *state, THClTensor *tensor) {
  int dims = THClTensor_nDimension(state, tensor);
  if(dims == 1) {
    int size0 = THClTensor_size(state, tensor, 0);
    for(int x = 0; x < size0; x++) {
      cout << THClTensor_get1d(state, tensor, x) << " ";
    }
    cout << endl;
  } else if(dims == 2) {
    int size0 = THClTensor_size(state, tensor, 0);
    int size1 = THClTensor_size(state, tensor, 1);
    for(int y = 0; y < size0; y++) {
      for(int x = 0; x < size1; x++) {
        cout << THClTensor_get2d(state, tensor, y, x) << " ";
      }
      cout << endl;
    }
  } else if(dims == 3) {
    int size0 = THClTensor_size(state, tensor, 0);
    for(int z = 0; z < size0; z++) {
      THClTensor *view = THClTensor_newSelect(state, tensor, 0, z);
      cout << "(" << z << ",.,.) =" << endl;
      printTensor(state, view);
      THClTensor_free(state, view);
    }
  } else if(dims == 4) {
    int size0 = THClTensor_size(state, tensor, 0);
    int size1 = THClTensor_size(state, tensor, 1);
    for(int x0 = 0; x0 < size0; x0++) {
      THClTensor *v0 = THClTensor_newSelect(state, tensor, 0, x0);
      for(int x1 = 0; x1 < size1; x1++) {
        THClTensor *v1 = THClTensor_newSelect(state, v0, 0, x1);
        cout << "(" << x0 << "," << x1 << ",.,.) =" << endl;
        printTensor(state, v1);
        THClTensor_free(state, v1);
      }
      THClTensor_free(state, v0);
    }
  } else {
    ostringstream oss;
    oss << "not implemented for " << dims << " dimensions";
    cout << oss.str() << endl;
    throw runtime_error(oss.str().c_str());
  }
}
PUBLIC VIRTUAL void Forward4::forward(THClState *state, THClTensor *input, THClTensor *weight, THClTensor *bias, THClTensor *output) {
  StatefulTimer::timeCheck("Forward4::forward start");

//  cout << "batchSize=" << conv->batchSize << endl;
  int numWorkgroups = conv->nOutputPlane * conv->batchSize * pixelsPerThread;
  int globalSize = workgroupSize * numWorkgroups;

//  cout << "input " << THClTensor_toString(state, input) << endl;
//  printTensor(state, input);
//  cout << "weight " << THClTensor_toString(state, weight) << endl;
//  printTensor(state, weight);
//  cout << "bias " << THClTensor_toString(state, bias) << endl;
//  printTensor(state, bias);

  THClKernels k(state, kernel);
  k.in((int)conv->batchSize);
  k.in(input);
  k.in(weight);
  k.out(output);
  k.localFloats((int)(conv->inputHeight * conv->inputWidth));
  k.localFloats((int)(conv->kH * conv->kW));

//  cout << "globalSize=" << globalSize << " workgroupSize=" << workgroupSize << endl;
  kernel->run_1d(globalSize, workgroupSize);
  StatefulTimer::timeCheck("Forward4::forward after call forward");

//  if(dim.biased) {
//  if(false){
    addBias->forward(
      conv->batchSize, conv->nOutputPlane, conv->outputHeight, conv->outputWidth,
      output, bias);
//  }

  EasyCL *cl = THClState_getClv2(state, device);
  cl->finish();

//  cout << "output " << THClTensor_toString(state, output) << endl;
//  printTensor(state, output);
}
PUBLIC Forward4::Forward4(THClState *state, int device, ClConvolver *conv)
      {
  this->state = state;
  this->device = device;
  this->conv = conv;

  if(conv->inputHeight != conv->inputWidth) {
    throw runtime_error("input must be square");
  }
  if(conv->kH != conv->kW) {
    throw runtime_error("filter must be square");
  }
  if(conv->dH != conv->dW) {
    throw runtime_error("stride must be 1");
  }
  if(conv->dH != 1) {
    throw runtime_error("stride must be 1");
  }
  if(conv->padH != conv->padW) {
    throw runtime_error("padH and padW must be same");
  }
  if(conv->padH != 0 && conv->padH != (conv->kH >> 1)) {
    throw runtime_error("padH must equal 0 or floor(kH/2)");
  }

  addBias = new AddBias(state, device, conv);

  workgroupSize = std::max(32, conv->outputHeight * conv->outputWidth); // no point in wasting threads....
  const int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;  // FIXME simplify this a bit :-P

  // see comments in forward4.cl,
  // if the outputimagesize * outputimagesize > maxWorkgroupSize,
  // then there wont be enough threads to process all the output points
  // of one image plane, so we create multiple workgroups for each
  // output image plane
  // here, we calculate how many workgroups we will need, in powers
  // of two:
  pixelsPerThread = 1;
  while(workgroupSize > maxWorkgroupSize) {
    workgroupSize = (workgroupSize + 1) >> 1;
    pixelsPerThread <<= 1;
  }

//  string uniqueName = __FILE__ ":forward4";
  ostringstream oss;
  oss << __FILE__ << ":forward4";
  oss << conv->inputHeight << "x" << conv->inputWidth;
  oss << "_" << conv->kH;
//  oss << "_" << conv->dH;
  oss << "_" << conv->padH;
  oss << "_" << conv->nInputPlane << "->" << conv->nOutputPlane;
  std::string uniqueName = oss.str();
  EasyCL *cl = THClState_getClv2(state, device);
  if(cl->kernelExists(uniqueName) ) {
    this->kernel = cl->getKernel(uniqueName);
    return;
  }

  int even = (conv->kH + 1) % 2;

  TemplatedKernel builder(cl);
  // kernelBuilder.set("", ""); // do this here...
  builder.set("filterSize", conv->kH);
  builder.set("filterSizeSquared", conv->kH * conv->kH);
  builder.set("even", even);
  builder.set("padding", conv->padH);
  builder.set("halfFilterSize", conv->kH >> 1);
  builder.set("nInputPlane", conv->nInputPlane);
  builder.set("nOutputPlane", conv->nOutputPlane);
  builder.set("inputSize", conv->inputHeight);
  builder.set("inputSizeSquared", conv->inputHeight * conv->inputWidth);
  builder.set("outputSize", conv->outputHeight);
  builder.set("outputSizeSquared", conv->outputHeight * conv->outputWidth);
  builder.set("workgroupSize", workgroupSize);
  builder.set("pixelsPerThread", pixelsPerThread);
  cout << "rendered template:\n" << builder.getRenderedKernel(getKernelTemplate()) << endl;
  kernel = builder.buildKernel(uniqueName, "Forward4.cl",
    getKernelTemplate(), "forward_4_by_n_outplane_smallercache");

  //cout << "workgroupSize=" << workgroupSize << " pixelsPerThread=" << pixelsPerThread << endl;

//  std::string options = "";
//  throw runtime_error("not implemented");
}

static std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel("kernel", "Forward4.cl")
  // ]]]
  // generated using cog, from Forward4.cl:
  const char * kernelSource =  
  "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
  "//\n" 
  "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
  "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
  "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
  "\n" 
  "#define gPixelsPerThread {{pixelsPerThread}}\n" 
  "#define gWorkgroupSize {{workgroupSize}}\n" 
  "#define gNumFilters {{nOutputPlane}}\n" 
  "#define gInputSize {{inputSize}}\n" 
  "#define gOutputSize {{outputSize}}\n" 
  "#define gFilterSize {{filterSize}}\n" 
  "#define gHalfFilterSize {{halfFilterSize}}\n" 
  "#define gPadding {{padding}}\n" 
  "#define gEven {{even}}\n" 
  "\n" 
  "//#define\n" 
  "//#define kH {{kH}}\n" 
  "//#define kW {{kW}}\n" 
  "//#define dH {{dH}}\n" 
  "//#define dW {{dW}}\n" 
  "//#define padH {{padH}}\n" 
  "//#define padW {{padW}}\n" 
  "#define gInputPlanes {{nInputPlane}}\n" 
  "\n" 
  "#define gInputSizeSquared {{inputSizeSquared}}\n" 
  "#define gOutputSizeSquared {{outputSizeSquared}}\n" 
  "#define gFilterSizeSquared {{filterSizeSquared}}\n" 
  "\n" 
  "void copyLocal(local float *target, global float const *source, int N) {\n" 
  "  int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n" 
  "  for (int loop = 0; loop < numLoops; loop++) {\n" 
  "    int offset = loop * get_local_size(0) + get_local_id(0);\n" 
  "    if (offset < N) {\n" 
  "      target[offset] = source[offset];\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "// workgroup id organized like: [n][filterid]\n" 
  "// local id organized like: [outrow][outcol]\n" 
  "// each thread iterates over: [inputplane][filterrow][filtercol]\n" 
  "// number workgroups = 32\n" 
  "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n" 
  "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n" 
  "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n" 
  "// output are organized like [n][filterid][outrow][outcol]\n" 
  "// the pixels per thread thing... :\n" 
  "// - we have one thread (~= cuda core) per output value,\n" 
  "//   ie one thread for each combination of [outrow][outcol]\n" 
  "// - however, the number of threads is typically limited on a gpu,\n" 
  "//   eg to 512 (eg Intel HD), or 1024 (eg nVidia K520)\n" 
  "// - so what happens if the number of output points is larger than\n" 
  "//   the maximum workgroup size?\n" 
  "// - then we have several possibilities really:\n" 
  "//   - we can divide the image into blocks, and process each block\n" 
  "//   separately.  This is probably a good option, but fair amount of\n" 
  "//   work\n" 
  "//   - we can get each thread to handle more than one output\n" 
  "//   pixel, by looping\n" 
  "//   - we can consider the output image in 1d, by putting the rows\n" 
  "//   one after another, and assign each contiguous workgroup-size\n" 
  "//   block to one workgroup\n" 
  "//   => this is how this kernel works\n" 
  "//   basically, it's a hack, so larger images actually run, without\n" 
  "//   crashing, and we can probably improve it a lot :-)\n" 
  "//\n" 
  "// So, when outputSize * outputSize > workgroupSize, then\n" 
  "// multiple workgroups will be created for each output plane\n" 
  "// the number of such workgroups is given by: `gPixelsPerThread`\n" 
  "// the id of our workgroup within such a set of workgroups is calculated\n" 
  "// as `pixel`\n" 
  "// effectiveLocalId is our local id if we had one enormous workgroup\n" 
  "// containing the whole output image plane\n" 
  "void kernel forward_4_by_n_outplane_smallercache(\n" 
  "      const int batchSize,\n" 
  "      global const float *images_data, int images_offset,\n" 
  "      global const float *filters_data, int filters_offset,\n" 
  "      global float *output_data, int output_offset,\n" 
  "      local float *_inputPlane,\n" 
  "      local float *_filterPlane\n" 
  "    ) {\n" 
  "  global const float *images = images_data + images_offset;\n" 
  "  global const float *filters = filters_data + filters_offset;\n" 
  "  global float *output = output_data + output_offset;\n" 
  "\n" 
  "  #define globalId (get_global_id(0))\n" 
  "\n" 
  "  #define localId (get_local_id(0))\n" 
  "  #define workgroupId (get_group_id(0))\n" 
  "//  const int workgroupSize = get_local_size(0);\n" 
  "  const int effectiveWorkgroupId = workgroupId / gPixelsPerThread;\n" 
  "  const int pixel = workgroupId % gPixelsPerThread;\n" 
  "  const int effectiveLocalId = localId + pixel * gWorkgroupSize;\n" 
  "  const int n = effectiveWorkgroupId / gNumFilters;\n" 
  "  const int outPlane = effectiveWorkgroupId % gNumFilters;\n" 
  "\n" 
  "  const int outputRow = effectiveLocalId / gOutputSize;\n" 
  "  const int outputCol = effectiveLocalId % gOutputSize;\n" 
  "\n" 
  "  if(get_global_id(0) == 0) {\n" 
  "//    output[0] = 0;\n" 
  "//    output[1] = 0;\n" 
  "  }\n" 
  "\n" 
  "  float sum = 0;\n" 
  "  for (int inputPlane = 0; inputPlane < gInputPlanes; inputPlane++) {\n" 
  "    barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "    copyLocal(_inputPlane, images + (n * gInputPlanes + inputPlane) * gInputSizeSquared, gInputSizeSquared);\n" 
  "    copyLocal(_filterPlane, filters + (outPlane * gInputPlanes + inputPlane) * gFilterSizeSquared, gFilterSizeSquared);\n" 
  "    barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "\n" 
  "    if (effectiveLocalId < gOutputSizeSquared) {\n" 
  "    if(get_global_id(0) == 0) {\n" 
  "//       output[0] = 122;\n" 
  "//      output[0] = output[0] * 10 + (u+2);\n" 
  "//        output[0] = gHalfFilterSize;\n" 
  "    }\n" 
  "      for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n" 
  "    if(get_global_id(0) == 0) {\n" 
  "//      output[0] = output[0] * 10 + (u+2);\n" 
  "//        output[0] = 124;\n" 
  "    }\n" 
  "        // trying to reduce register pressure...\n" 
  "        #if gPadding != 0\n" 
  "        #define inputRow (outputRow + u)\n" 
  "        #else\n" 
  "        #define inputRow (outputRow + u + gHalfFilterSize)\n" 
  "        #endif\n" 
  "        int inputimagerowoffset = inputRow * gInputSize;\n" 
  "        int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
  "        bool rowOk = inputRow >= 0 && inputRow < gInputSize;\n" 
  "        for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n" 
  "          #if gPadding != 0\n" 
  "          #define inputCol (outputCol + v)\n" 
  "          #else\n" 
  "          #define inputCol (outputCol + v + gHalfFilterSize)\n" 
  "          #endif\n" 
  "          bool process = rowOk && inputCol >= 0 && inputCol < gInputSize;\n" 
  "          if(get_global_id(0) == 0) {\n" 
  "            output[1] = output[1] * 10 + (v+2);\n" 
  "            output[2+(u+1)*2+(v+1)] = _inputPlane[ (u+1)*2+(v+1)];\n" 
  "          }\n" 
  "          if (process) {\n" 
  "              sum += _inputPlane[ inputimagerowoffset + inputCol] * _filterPlane[ filterrowoffset + v ];\n" 
  "          }\n" 
  "        }\n" 
  "      }\n" 
  "    }\n" 
  "  }\n" 
  "  // output are organized like [imageid][filterid][row][col]\n" 
  "  #define resultIndex (( n * gNumFilters + outPlane) * gOutputSizeSquared + effectiveLocalId)\n" 
  "  if (effectiveLocalId < gOutputSizeSquared) {\n" 
  "    output[resultIndex ] = sum;\n" 
  "  }\n" 
  "  if(get_global_id(0) == 0) {\n" 
  "//    output[0] = filters[0];\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

