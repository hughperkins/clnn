// from SpatialMaxPooling.cu:

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

std::string SpatialMaxPooling_getKernelTemplate();

static int clnn_SpatialMaxPooling_updateOutput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THClTensor *output = (THClTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  THClTensor *indices = (THClTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.ClTensor");

//  float *indices_data;
//  float *output_data;
//  float *input_data;

  THAssert(THClTensor_checkGPU(state, 3, input, output, indices));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = floor(float(nInputCols - kW) / float(dW) + 1);
    long nOutputRows = floor(float(nInputRows - kH) / float(dH) + 1);

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THClTensor_newContiguous(state, input);
//    input_data = THClTensor_data(state, input);

    THClTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    THClTensor_resize4d(state, indices, 2, nInputPlane, nOutputRows, nOutputCols);

//    indices_data = THClTensor_data(state, indices);
//    output_data = THClTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
//    maxpool <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//      input_data, output_data,
//      indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
//      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    //THError("not implemented");
    EasyCL *cl = input->storage->cl;
    std::string uniqueName = __FILE__ "maxpool";
    CLKernel *kernel = 0;
    if(cl->kernelExists(uniqueName)) {
      kernel = cl->getKernel(uniqueName);
    } else {
      TemplatedKernel kernelBuilder(cl);
      kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
        SpatialMaxPooling_getKernelTemplate(), "maxpool");
    }

    THClKernels k(state, kernel);
    k.in(input);
    k.out(output);
    k.out(indices);
    k.in((int)(nInputPlane*nOutputCols*nOutputRows));
    k.in((int)0);
    k.in((int)nInputPlane);
    k.in((int)nInputRows);
    k.in((int)nInputCols);
    k.in((int)kH);
    k.in((int)kW);
    k.in((int)dH);
    k.in((int)dW);
    k.run(blocks, threads);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = floor(float(nInputCols - kW) / float(dW) + 1);
    long nOutputRows = floor(float(nInputRows - kH) / float(dH) + 1);

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THClTensor_newContiguous(state, input);
//    input_data = THClTensor_data(state, input);

    THClTensor_resize4d(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    THClTensor_resize5d(state, indices, 2, nbatch, nInputPlane, nOutputRows, nOutputCols);

//    indices_data = THClTensor_data(state, indices);
//    output_data = THClTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel

    EasyCL *cl = input->storage->cl;
    std::string uniqueName = __FILE__ "maxpool";
    CLKernel *kernel = 0;
    if(cl->kernelExists(uniqueName)) {
      kernel = cl->getKernel(uniqueName);
    } else {
      TemplatedKernel kernelBuilder(cl);
      kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
        SpatialMaxPooling_getKernelTemplate(), "maxpool");
    }

    THClKernels k(state, kernel);
    k.in(input);
    k.out(output);
    k.out(indices);
    k.in((int)(nbatch*nInputPlane*nOutputCols*nOutputRows));
    k.in((int)0);
    k.in((int)nInputPlane);
    k.in((int)nInputRows);
    k.in((int)nInputCols);
    k.in((int)kH);
    k.in((int)kW);
    k.in((int)dH);
    k.in((int)dW);
    k.run(blocks, threads);

//    maxpool <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//      input_data, output_data,
//      indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
//      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
//    THError("not implemented");
  }

  // clean
  THClTensor_free(state, input);

  return 1;
}

static int clnn_SpatialMaxPooling_updateGradInput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  bool atomic = (dW != kW) || (dH != kH);
//  cout << "atomic=" << atomic << " kW=" << kW << " dW=" << dW << " kH=" << kH << " dH=" << dH << endl;
//  cout << "input->nDimension=" << input->nDimension << endl;

  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  THClTensor *indices = (THClTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.ClTensor");

//  float *indices_data;
//  float *gradInput_data;
//  float *gradOutput_data;

  THAssert(THClTensor_checkGPU(state, 4, input, gradOutput, indices, gradInput));

  input = THClTensor_newContiguous(state, input);
  gradOutput = THClTensor_newContiguous(state, gradOutput);

  if (input->nDimension == 3) {
//    long nInputCols = input->size[2];
//    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
//    long nOutputCols = gradOutput->size[2];
//    long nOutputRows = gradOutput->size[1];

    THClTensor_resizeAs(state, gradInput, input);
    THClTensor_zero(state, gradInput);

//    indices_data = THClTensor_data(state, indices);
//    gradOutput_data = THClTensor_data(state, gradOutput);
//    gradInput_data = THClTensor_data(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
//      atomicmaxgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//        gradInput_data, gradOutput_data,
//        indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
//        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
      THError("not implemented");
    }
    else
    {
      // run updateGradInput kernel
//      atomicmaxgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//        gradInput_data, gradOutput_data,
//        indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
//        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
      THError("not implemented");
    }
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = gradOutput->size[3];
    long nOutputRows = gradOutput->size[2];

    THClTensor_resizeAs(state, gradInput, input);
    THClTensor_zero(state, gradInput);

//    indices_data = THClTensor_data(state, indices);
//    gradOutput_data = THClTensor_data(state, gradOutput);
//    gradInput_data = THClTensor_data(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
//      atomicmaxgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//        gradInput_data, gradOutput_data,
//        indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
//        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
      THError("not implemented");
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically

      EasyCL *cl = input->storage->cl;
      std::string uniqueName = __FILE__ "maxgradinput";
      CLKernel *kernel = 0;
      if(cl->kernelExists(uniqueName)) {
        kernel = cl->getKernel(uniqueName);
      } else {
        TemplatedKernel kernelBuilder(cl);
        kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
          SpatialMaxPooling_getKernelTemplate(), "maxgradinput");
      }

      THClKernels k(state, kernel);
      k.out(gradInput);
      k.in(gradOutput);
      k.in(indices);
      k.in((int)(nbatch*nInputPlane*nOutputCols*nOutputRows));
      k.in((int)0);

      k.in((int)nInputPlane);
      k.in((int)nInputRows);
      k.in((int)nInputCols);

      k.in((int)kH);
      k.in((int)kW);
      k.in((int)dH);
      k.in((int)dW);

      k.run(blocks, threads);

//      maxgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//        gradInput_data, gradOutput_data,
//        indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
//        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
//      THError("not implemented");
    }
  }

  // clean
  THClTensor_free(state, input);
  THClTensor_free(state, gradOutput);

  return 1;
}

static const struct luaL_Reg clnn_SpatialMaxPooling__ [] = {
  {"SpatialMaxPooling_updateOutput", clnn_SpatialMaxPooling_updateOutput},
  {"SpatialMaxPooling_updateGradInput", clnn_SpatialMaxPooling_updateGradInput},
  {NULL, NULL}
};

void clnn_SpatialMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialMaxPooling__, "nn");
  lua_pop(L,1);
}

std::string SpatialMaxPooling_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "SpatialMaxPooling.cl" )
  // ]]]
  // generated using cog, from SpatialMaxPooling.cl:
  const char * kernelSource =  
  "// from SpatialMaxPooling.cu:\n" 
  "\n" 
  "/*\n" 
  " * Description:\n" 
  " *    this function maxpools an input 4D tensor along dimensions 2 and 3\n" 
  " *    4D input, 4D output, 4D argmax x and y\n" 
  " */\n" 
  "kernel void maxpool(const global float *input_data, int input_offset,\n" 
  "    global float *output_data, int output_offset,\n" 
  "    global float *indices_data, int indices_offset,\n" 
  "    int indices_x_offset,\n" 
  "    int indices_y_offset,\n" 
  "    int input_n, int input_h, int input_w,\n" 
  "    int kH, int kW, int dH, int dW)\n" 
  "{\n" 
  "  global const float *input = input_data + input_offset;\n" 
  "  global float *output = output_data + output_offset;\n" 
  "  global float *indices_x = indices_data + indices_offset + indices_x_offset;\n" 
  "  global float *indices_y = indices_data + indices_offset + indices_y_offset;\n" 
  "\n" 
  "  // iterators\n" 
  "  int xx, yy;\n" 
  "\n" 
  "  // output size\n" 
  "  const int output_w = (input_w - kW) / dW + 1;\n" 
  "  const int output_h = (input_h - kH) / dH + 1;\n" 
  "\n" 
  "  // compute offsets based on thread/block ID\n" 
  "  int o = get_group_id(0);\n" 
  "  int i = o;\n" 
  "  //int k = get_group_id(0) % input_n;\n" 
  "\n" 
  "  int xx_start = get_local_id(0);\n" 
  "  int xx_end = output_w;\n" 
  "  const int xx_step = get_local_size(0);\n" 
  "\n" 
  "  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);\n" 
  "  int yy_end = output_h;\n" 
  "  const int yy_step = get_local_size(1)*get_num_groups(1);\n" 
  "\n" 
  "  // select input/output plane\n" 
  "  output = output + o*output_w*output_h;\n" 
  "  input = input + i*input_w*input_h;\n" 
  "  indices_x = indices_x + o*output_w*output_h;\n" 
  "  indices_y = indices_y + o*output_w*output_h;\n" 
  "\n" 
  "  // For all output pixels...\n" 
  "  for(yy = yy_start; yy < yy_end; yy+=yy_step) {\n" 
  "    for(xx = xx_start; xx < xx_end; xx+=xx_step) {\n" 
  "      // Compute the mean of the input image...\n" 
  "      global const float *ptr_input = input + yy*dH*input_w + xx*dW;\n" 
  "      global float *ptr_output = output + yy*output_w + xx;\n" 
  "      global float *ptr_ind_x = indices_x + yy*output_w + xx;\n" 
  "      global float *ptr_ind_y = indices_y + yy*output_w + xx;\n" 
  "      int argmax_x = -1;\n" 
  "      int argmax_y = -1;\n" 
  "      float max = -FLT_MAX;\n" 
  "      int kx, ky;\n" 
  "      for(ky = 0; ky < kH; ky++) {\n" 
  "        for(kx = 0; kx < kW; kx++) {\n" 
  "          float val = ptr_input[kx];\n" 
  "          if (val > max) {\n" 
  "            max = val;\n" 
  "            argmax_x = kx;\n" 
  "            argmax_y = ky;\n" 
  "          }\n" 
  "        }\n" 
  "        ptr_input += input_w; // next input line\n" 
  "      }\n" 
  "      // Update output and argmax\n" 
  "      *ptr_output = max;\n" 
  "      *ptr_ind_x = argmax_x + 1;\n" 
  "      *ptr_ind_y = argmax_y + 1;\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "/*\n" 
  " * Description:\n" 
  " *    this function computes the gradInput from weight and gradOutput\n" 
  " */\n" 
  "kernel void maxgradinput(global float *gradInput_data, int gradInput_offset,\n" 
  "    global const float *gradOutput_data, int gradOutput_offset,\n" 
  "    global const float *indices_data, int indices_offset,\n" 
  "    int indices_x_offset, int indices_y_offset,\n" 
  "   int input_n, int input_h, int input_w,\n" 
  "   int kH, int kW, int dH, int dW)\n" 
  "{\n" 
  "  global float *gradInput = gradInput_data + gradInput_offset;\n" 
  "  global const float *gradOutput = gradOutput_data + gradOutput_offset;\n" 
  "  global const float *indices_x = indices_data + indices_offset + indices_x_offset;\n" 
  "  global const float *indices_y = indices_data + indices_offset + indices_y_offset;\n" 
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
  "  //int k = get_group_id(0) % input_n;\n" 
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
  "  indices_x = indices_x + o*output_w*output_h;\n" 
  "  indices_y = indices_y + o*output_w*output_h;\n" 
  "\n" 
  "  // compute gradInput\n" 
  "  for(yy = yy_start; yy < yy_end; yy+=yy_step) {\n" 
  "    for(xx = xx_start; xx < xx_end; xx+=xx_step) {\n" 
  "      global float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;\n" 
  "      global const float *ptr_gradOutput = gradOutput + yy*output_w + xx;\n" 
  "      global const float *ptr_ind_x = indices_x + yy*output_w + xx;\n" 
  "      global const float *ptr_ind_y = indices_y + yy*output_w + xx;\n" 
  "      float z = *ptr_gradOutput;\n" 
  "\n" 
  "      int argmax_x = (*ptr_ind_x)-1;\n" 
  "      int argmax_y = (*ptr_ind_y)-1;\n" 
  "\n" 
  "      ptr_gradInput[argmax_x + argmax_y*input_w] += z;\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "/*\n" 
  " * Description:\n" 
  " *    this function computes the gradInput from weight and gradOutput\n" 
  " *    when kH != dH or kW != dW (uses atomic add)\n" 
  " */\n" 
  "//kernel void atomicmaxgradinput(\n" 
  "//  global float *gradInput_data, int gradInput_offset,\n" 
  "//  global float *gradOutput_data, int gradOutput_offset,\n" 
  "//  global float *indices_data, int indices_offset,\n" 
  "//  int indices_x_offset, int indices_y_offset,\n" 
  "//  int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW\n" 
  "//)\n" 
  "//{\n" 
  "//  global float *gradInput = gradInput_data + gradInput_offset;\n" 
  "//  global float *gradOutput = gradOutput_data + gradOutput_offset;\n" 
  "//  global float *indices_x = indices_data + indices_offset + indices_x_offset;\n" 
  "//  global float *indices_y = indices_data + indices_offset + indices_y_offset;\n" 
  "\n" 
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
  "//  //int k = get_group_id(0) % input_n;\n" 
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
  "//  indices_x = indices_x + o*output_w*output_h;\n" 
  "//  indices_y = indices_y + o*output_w*output_h;\n" 
  "\n" 
  "//  // compute gradInput\n" 
  "//  for(yy = yy_start; yy < yy_end; yy+=yy_step) {\n" 
  "//    for(xx = xx_start; xx < xx_end; xx+=xx_step) {\n" 
  "//      global float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;\n" 
  "//      global float *ptr_gradOutput = gradOutput + yy*output_w + xx;\n" 
  "//      global float *ptr_ind_x = indices_x + yy*output_w + xx;\n" 
  "//      global float *ptr_ind_y = indices_y + yy*output_w + xx;\n" 
  "//      float z = *ptr_gradOutput;\n" 
  "\n" 
  "//      int argmax_x = (*ptr_ind_x)-1;\n" 
  "//      int argmax_y = (*ptr_ind_y)-1;\n" 
  "\n" 
  "//      // atomic add since different threads could update same variable\n" 
  "////      atomicAdd(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);\n" 
  "//      // hmmm, this doesnt work with float :-(  need another way...\n" 
  "//      atomic_add(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);\n" 
  "//    }\n" 
  "//  }\n" 
  "//}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

