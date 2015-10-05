// from SpatialConvolutionMM.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"
#include "DeviceInfo.h"
#include "EasyCL.h"
#include "conv/ClConvolver.h"
#include "conv/Forward.h"
#include "conv/BackGradIm2Col.h"
#include "conv/GradWeightsIm2Col.h"

#include <iostream>
#include <string>
using namespace std;

static int clnn_SpatialConvolutionMM_updateOutput(lua_State *L) {
//  cout << "clnn_SpatialConvolutionMM_updateOutput(lua_State *L)" << endl;
  THClState *state = getCltorchState(L);
  // Input
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  // Convolver
  lua_getfield(L, 1, "convolver");
  ClConvolver *conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
//  , "nn.ClConvolver")
  if(conv == 0) {
    ClConvolver_new(L);
    lua_pushvalue(L, -1);
    lua_setfield(L, 1, "convolver");
    conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
  }
  if(conv == 0) {
    THError("failed to create convolver object");
  }

  // Params:
  conv->dW = luaT_getfieldcheckint(L, 1, "dW");
  conv->dH = luaT_getfieldcheckint(L, 1, "dH");
  conv->kW = luaT_getfieldcheckint(L, 1, "kW");
  conv->kH = luaT_getfieldcheckint(L, 1, "kH");
  conv->nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  conv->nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  conv->padW = luaT_getfieldcheckint(L, 1, "padW");
  conv->padH = luaT_getfieldcheckint(L, 1, "padH");

  conv->batch = 1;
  if (input->nDimension == 3) {
    luaL_argcheck(L, input->size[0] == conv->nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    conv->batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
  } else {
    luaL_argcheck(L, input->size[1] == conv->nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  conv->inputWidth   = input->size[3];
  conv->inputHeight  = input->size[2];
  conv->outputWidth  = (conv->inputWidth + 2*conv->padW - conv->kW) / conv->dW + 1;
  conv->outputHeight = (conv->inputHeight + 2*conv->padH - conv->kH) / conv->dH + 1;

  if (conv->outputWidth < 1 || conv->outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        conv->nInputPlane,conv->inputHeight,conv->inputWidth,conv->nOutputPlane,conv->outputHeight,conv->outputWidth);

  if(conv->forwarder == 0) {
    conv->forwarder = Forward::instance(state, input->storage->device, conv);
  }

  THClTensor *weight = (THClTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.ClTensor");
  THClTensor *bias = (THClTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.ClTensor");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 4, input, output, weight,
                                 bias));

  // Batch size + input planes
  conv->batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, output, conv->batchSize, conv->nOutputPlane, conv->outputHeight, conv->outputWidth);

  conv->forwarder->forward(state, input, weight, bias, output);

  // return output
  return 1;
}

static int clnn_SpatialConvolutionMM_updateGradInput(lua_State *L) {
  THClState *state = getCltorchState(L);
  // Inputs
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");

  lua_getfield(L, 1, "convolver");
  ClConvolver *conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
  if(conv == 0) {
    THError("clconvolver object not found");
  }

  THClTensor *weight = (THClTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.ClTensor");
  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 4, input, gradOutput, weight,
                                 gradInput));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  conv->batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    conv->batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  if(conv->backGradder == 0) {
    conv->backGradder = new BackGradIm2Col(state, input->storage->device, conv);
  }

  // Resize output
  THClTensor_resize4d(state, gradInput, conv->batchSize, conv->nInputPlane, conv->inputHeight, conv->inputWidth);

  conv->backGradder->updateGradInput(state, input, gradOutput, weight, gradInput);

  // Return gradInput
  return 1;
}

static int clnn_SpatialConvolutionMM_accGradParameters(lua_State *L) {
  THClState *state = getCltorchState(L);
  // Inputs
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");

  lua_getfield(L, 1, "convolver");
  ClConvolver *conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
  if(conv == 0) {
    THError("clconvolver object not found");
  }

  // Params
  float scale = luaL_optnumber(L, 4, 1);

  THClTensor *gradWeight = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.ClTensor");
  THClTensor *gradBias = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 4, input, gradOutput, gradWeight,
                                 gradBias));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  conv->batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    conv->batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  // Batch size + input planes
  conv->batchSize = input->size[0];

  if(conv->gradWeights == 0) {
    conv->gradWeights = new GradWeightsIm2Col(state, input->storage->device, conv);
  }

  conv->gradWeights->accGradParameters(state, input, gradOutput, gradWeight, gradBias, scale);

  // Return nothing
  return 0;
}

static const struct luaL_Reg clnn_SpatialConvolutionMM__ [] = {
  {"SpatialConvolutionMM_updateOutput", clnn_SpatialConvolutionMM_updateOutput},
  {"SpatialConvolutionMM_updateGradInput", clnn_SpatialConvolutionMM_updateGradInput},
  {"SpatialConvolutionMM_accGradParameters", clnn_SpatialConvolutionMM_accGradParameters},
  {NULL, NULL}
};

void clnn_SpatialConvolutionMM_init(lua_State *L)
{
//  cout << "registering spatialconvolutionmm" << endl;
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialConvolutionMM__, "nn");
  lua_pop(L,1);
}

