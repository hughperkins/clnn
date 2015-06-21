// from SpatialMaxPooling.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"

#include <iostream>
#include <string>
using namespace std;

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

  float *indices_data;
  float *output_data;
  float *input_data;

  THAssert(THClTensor_checkGPU(state, 3, input, output, indices));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THClTensor_newContiguous(state, input);
    input_data = THClTensor_data(state, input);

    THClTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    THClTensor_resize4d(state, indices, 2, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THClTensor_data(state, indices);
    output_data = THClTensor_data(state, output);

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
    THError("not implemented");
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THClTensor_newContiguous(state, input);
    input_data = THClTensor_data(state, input);

    THClTensor_resize4d(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    THClTensor_resize5d(state, indices, 2, nbatch, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THClTensor_data(state, indices);
    output_data = THClTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
//    maxpool <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//      input_data, output_data,
//      indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
//      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    THError("not implemented");
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

  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  THClTensor *indices = (THClTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.ClTensor");

  float *indices_data;
  float *gradInput_data;
  float *gradOutput_data;

  THAssert(THClTensor_checkGPU(state, 4, input, gradOutput, indices, gradInput));

  input = THClTensor_newContiguous(state, input);
  gradOutput = THClTensor_newContiguous(state, gradOutput);

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = gradOutput->size[2];
    long nOutputRows = gradOutput->size[1];

    THClTensor_resizeAs(state, gradInput, input);
    THClTensor_zero(state, gradInput);

    indices_data = THClTensor_data(state, indices);
    gradOutput_data = THClTensor_data(state, gradOutput);
    gradInput_data = THClTensor_data(state, gradInput);

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

    indices_data = THClTensor_data(state, indices);
    gradOutput_data = THClTensor_data(state, gradOutput);
    gradInput_data = THClTensor_data(state, gradInput);

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
//      maxgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//        gradInput_data, gradOutput_data,
//        indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
//        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
      THError("not implemented");
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

