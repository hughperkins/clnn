// from SpatialAveragePooling.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"

#include "EasyCL.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"

#include <iostream>
#include <string>
using namespace std;

static std::string getKernelTemplate();

static int clnn_SpatialAveragePooling_updateOutput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  bool ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");
  bool count_include_pad = luaT_getfieldcheckboolean(L, 1, "count_include_pad");

  THClTensor *output = (THClTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 2, input, output));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  luaL_argcheck(L, nInputCols >= kW - padW && nInputRows >= kH - padH, 2, "input image smaller than kernel size");
  luaL_argcheck(L, kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }

  input = THClTensor_newContiguous(state, input);
  // input_data = THClTensor_data(state, input);

  THClTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  
  // float* output_data = THClTensor_data(state, output);

  int count = THClTensor_nElement(state, output);

  EasyCL *cl = input->storage->cl;
  const char *uniqueName = 0;
  if(count_include_pad) {
    uniqueName = __FILE__ "forward_includepad";
  } else {
    uniqueName = __FILE__ "forward_not_includepad";
  }
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("forward", 1);
    kernelBuilder.set("Dtype", "float");
    kernelBuilder.set("COUNT_INCLUDE_PAD", count_include_pad);
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      getKernelTemplate(), "subsample");
  }

  THClKernels k(state, kernel);
  k.in(int)count);
  k.in(input);
  k.in((int)batchSize);
  k.in((int)nInputPlane);
  k.in((int)nInputRows);
  k.in((int)nInputCols);
  k.in((int)nOutputRows);
  k.in((int)nOutputCols);
  k.in((int)kH);
  k.in((int)kW);
  k.in((int)dH);
  k.in((int)dW);
  k.in((int)padH);
  k.in((int)padW);
  k.out(output);

  dim3 blocks(GET_BLOCKS(state, count));
  dim3 threads(GET_CL_NUM_THREADS(state));
  k.run(blocks, threads);

//    AvePoolForward<float, true>
//      <<<GET_BLOCKS(count), CL_NUM_THREADS, 0, THClState_getCurrentStream(state) >>>(
//        count, input_data,
//        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
//        kH, kW, dH, dW, padH, padW, output_data);

  if(input->nDimension == 3)
    THClTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

  THClTensor_free(state, input);

  return 1;
}

static int clnn_SpatialAveragePooling_updateGradInput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  THAssert(THClTensor_checkGPU(state, 3, input, gradInput, gradOutput));

  long nInputCols = 0;
  long nInputRows = 0;
  long nInputPlane = 0;
  long nbatch = 0;
  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    nbatch = 1;
  } else {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    nbatch = input->size[0];
  }

  // float *gradOutput_data = THClTensor_data(state, gradOutput);
  // float *gradInput_data;

  THClTensor_resizeAs(state, gradInput, input);
  THClTensor_zero(state, gradInput);
  // gradInput_data = THClTensor_data(state, gradInput);

  int yblocks = (int)(16L / nInputPlane);
  yblocks = yblocks < 1 ? 1 : yblocks;
  dim3 blocks(nInputPlane*nbatch,yblocks);
  dim3 threads(32,8);

  // run updateGradInput kernel
  if ((kH == dH && kW == dW) || (kH == nInputRows && kW == nInputCols)) {
    EasyCL *cl = input->storage->cl;
    std::string uniqueName = __FILE__ "subgradinput";
    CLKernel *kernel = 0;
    if(cl->kernelExists(uniqueName)) {
      kernel = cl->getKernel(uniqueName);
    } else {
      TemplatedKernel kernelBuilder(cl);
      kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
        getKernelTemplate(), "subgradinput");
    }

    THClKernels k(state, kernel);
    k.out(gradInput);
    k.in(gradOutput);

    k.in((int)nInputPlane);
    k.in((int)nInputRows);
    k.in((int)nInputCols);

    k.in((int)kH);
    k.in((int)kW);
    k.in((int)dH);
    k.in((int)dW);

    k.run(blocks, threads);

//      subgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
//                                          nInputPlane, nInputRows, nInputCols,
//                                          kH, kW, dH, dW);
  } else {
    THError("Not implemented");
//      subgradinputAtomic <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
//                                                nInputPlane, nInputRows, nInputCols,
//                                                kH, kW, dH, dW);
  }

  return 1;
}

static const struct luaL_Reg clnn_SpatialAveragePooling__ [] = {
  {"SpatialAveragePooling_updateOutput", clnn_SpatialAveragePooling_updateOutput},
  {"SpatialAveragePooling_updateGradInput", clnn_SpatialAveragePooling_updateGradInput},
  {NULL, NULL}
};

void clnn_SpatialAveragePooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialAveragePooling__, "nn");
  lua_pop(L,1);
}

//#undef CL_MAX_THREADS

static std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "SpatialAveragePooling.cl" )
  // ]]]
  // generated using cog, from SpatialAveragePooling.cl:
  const char * kernelSource =  
  "// from SpatialAveragePooling.cu:\n" 
  "\n" 
  "/*\n" 
  " * Description:\n" 
  " *    this function avg-pools an input 3D tensor along dimensions 1 and 2\n" 
  " *    3D input, 3D output\n" 
  " */\n" 
  "kernel void subsample(\n" 
  "  const global float *input_data, int input_offset,\n" 
  "  global float *output_data, int output_offset,\n" 
  "  int input_n, int input_h, int input_w,\n" 
  "  int kH, int kW, int dH, int dW)\n" 
  "{\n" 
  "  global const float *input = input_data + input_offset;\n" 
  "  global float *output = output_data + output_offset;\n" 
  "\n" 
  "  // iterators\n" 
  "  int xx, yy;\n" 
  "\n" 
  "  // output size\n" 
  "  int output_w = (input_w - kW) / dW + 1;\n" 
  "  int output_h = (input_h - kH) / dH + 1;\n" 
  "\n" 
  "  // compute offsets based on thread/block ID\n" 
  "  int o = get_group_id(0);\n" 
  "  int i = o;\n" 
  "\n" 
  "  int xx_start = get_local_id(0);\n" 
  "  int xx_end = output_w;\n" 
  "  int xx_step = get_local_size(0);\n" 
  "\n" 
  "  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);\n" 
  "  int yy_end = output_h;\n" 
  "  int yy_step = get_local_size(1)*get_num_groups(1);\n" 
  "\n" 
  "  // select input/output plane\n" 
  "  output = output + o*output_w*output_h;\n" 
  "  input = input + i*input_w*input_h;\n" 
  "\n" 
  "  // For all output pixels...\n" 
  "  for(yy = yy_start; yy < yy_end; yy+=yy_step) {\n" 
  "    for(xx = xx_start; xx < xx_end; xx+=xx_step) {\n" 
  "      // Compute the mean of the input image...\n" 
  "      const global float *ptr_input = input + yy*dH*input_w + xx*dW;\n" 
  "      global float *ptr_output = output + yy*output_w + xx;\n" 
  "      float sum = 0;\n" 
  "      int kx, ky;\n" 
  "      for(ky = 0; ky < kH; ky++) {\n" 
  "        for(kx = 0; kx < kW; kx++)\n" 
  "          sum += ptr_input[kx];\n" 
  "        ptr_input += input_w; // next input line\n" 
  "      }\n" 
  "      // Update output\n" 
  "      *ptr_output = sum/(float)(kW*kH);\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "/*\n" 
  " * Description:\n" 
  " *    this function computes the gradInput from gradOutput\n" 
  " */\n" 
  "kernel void subgradinput(\n" 
  "  global float *gradInput_data, int gradInput_offset,\n" 
  "  const global float *gradOutput_data, int gradOutput_offset,\n" 
  "  int input_n, int input_h, int input_w,\n" 
  "  int kH, int kW, int dH, int dW)\n" 
  "{\n" 
  "  global float *gradInput = gradInput_data + gradInput_offset;\n" 
  "  const global float *gradOutput = gradOutput_data + gradOutput_offset;\n" 
  "\n" 
  "  // iterators\n" 
  "  int xx, yy;\n" 
  "\n" 
  "  // output size\n" 
  "  int output_w = (input_w - kW) / dW + 1;\n" 
  "  int output_h = (input_h - kH) / dH + 1;\n" 
  "\n" 
  "  // compute offsets based on thread/block ID\n" 
  "  int o = get_group_id(0);\n" 
  "  int i = o;\n" 
  "\n" 
  "  int xx_start = get_local_id(0);\n" 
  "  int xx_end = output_w;\n" 
  "  int xx_step = get_local_size(0);\n" 
  "\n" 
  "  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);\n" 
  "  int yy_end = output_h;\n" 
  "  int yy_step = get_local_size(1)*get_num_groups(1);\n" 
  "\n" 
  "  // select input/output plane\n" 
  "  gradOutput = gradOutput + o*output_w*output_h;\n" 
  "  gradInput = gradInput + i*input_w*input_h;\n" 
  "\n" 
  "  // compute gradInput\n" 
  "  for(yy = yy_start; yy < yy_end; yy+=yy_step) {\n" 
  "    for(xx = xx_start; xx < xx_end; xx+=xx_step) {\n" 
  "      global float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;\n" 
  "      const global float *ptr_gradOutput = gradOutput + yy*output_w + xx;\n" 
  "      float z = *ptr_gradOutput;\n" 
  "      int kx, ky;\n" 
  "      for(ky = 0; ky < kH; ky++) {\n" 
  "        for(kx = 0; kx < kW; kx++)\n" 
  "          ptr_gradInput[kx] += z / (float)(kW*kH);\n" 
  "        ptr_gradInput += input_w;\n" 
  "      }\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "/*\n" 
  " * Description:\n" 
  " *    this function computes the gradInput from gradOutput\n" 
  " *    but with an atomic accumulation. It is needed to be done so\n" 
  " *    for cases of kH != dH and kW != dW\n" 
  " */\n" 
  "//kernel void subgradinputAtomic(float *gradInput, float *gradOutput,\n" 
  "//                                   int input_n, int input_h, int input_w,\n" 
  "//                                   int kH, int kW, int dH, int dW)\n" 
  "//{\n" 
  "//  // iterators\n" 
  "//  int xx, yy;\n" 
  "\n" 
  "//  // output size\n" 
  "//  int output_w = (input_w - kW) / dW + 1;\n" 
  "//  int output_h = (input_h - kH) / dH + 1;\n" 
  "\n" 
  "//  // compute offsets based on thread/block ID\n" 
  "//  int o = get_group_id(0);\n" 
  "//  int i = o;\n" 
  "\n" 
  "//  int xx_start = get_local_id(0);\n" 
  "//  int xx_end = output_w;\n" 
  "//  int xx_step = get_local_size(0);\n" 
  "\n" 
  "//  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);\n" 
  "//  int yy_end = output_h;\n" 
  "//  int yy_step = get_local_size(1)*get_num_groups(1);\n" 
  "\n" 
  "//  // select input/output plane\n" 
  "//  gradOutput = gradOutput + o*output_w*output_h;\n" 
  "//  gradInput = gradInput + i*input_w*input_h;\n" 
  "\n" 
  "//  // compute gradInput\n" 
  "//  for(yy = yy_start; yy < yy_end; yy+=yy_step) {\n" 
  "//    for(xx = xx_start; xx < xx_end; xx+=xx_step) {\n" 
  "//      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;\n" 
  "//      float *ptr_gradOutput = gradOutput + yy*output_w + xx;\n" 
  "//      float z = *ptr_gradOutput;\n" 
  "//      int kx, ky;\n" 
  "//      for(ky = 0; ky < kH; ky++) {\n" 
  "//        for(kx = 0; kx < kW; kx++) {\n" 
  "//          atomicAdd(&(ptr_gradInput[kx]), z / float(kW*kH));\n" 
  "//        }\n" 
  "//        ptr_gradInput += input_w;\n" 
  "//      }\n" 
  "//    }\n" 
  "//  }\n" 
  "//}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}
