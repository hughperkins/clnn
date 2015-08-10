// from SpatialAveragePooling.cu:

#include "utils.h"

//#define CL_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

static int clnn_SpatialAveragePooling_updateOutput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THClTensor *output = (THClTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  THAssert(THClTensor_checkGPU(state, 2, input, output));

  float *output_data;
  float *input_data;

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    long nInputPlane = input->size[0];

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THClTensor_newContiguous(state, input);
    input_data = THClTensor_data(state, input);

    THClTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    output_data = THClTensor_data(state, output);

    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    THError("Not implemented");
//    subsample <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (input_data, output_data,
//                                     nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    long nInputPlane = input->size[1];

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THClTensor_newContiguous(state, input);
    input_data = THClTensor_data(state, input);

    THClTensor_resize4d(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    output_data = THClTensor_data(state, output);

    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    THError("Not implemented");
//    subsample <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (input_data, output_data,
//                                     nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  }

  // clean
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

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];

    float *gradOutput_data = THClTensor_data(state, gradOutput);
    float *gradInput_data;

    THClTensor_resizeAs(state, gradInput, input);
    THClTensor_zero(state, gradInput);
    gradInput_data = THClTensor_data(state, gradInput);

    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kH == dH && kW == dW) {
      THError("Not implemented");
//      subgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
//                                          nInputPlane, nInputRows, nInputCols,
//                                          kH, kW, dH, dW);
    } else {
      THError("Not implemented");
//      subgradinputAtomic <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
//                                                nInputPlane, nInputRows, nInputCols,
//                                                kH, kW, dH, dW);
    }
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];

    float *gradOutput_data = THClTensor_data(state, gradOutput);
    float *gradInput_data;

    THClTensor_resizeAs(state, gradInput, input);
    THClTensor_zero(state, gradInput);
    gradInput_data = THClTensor_data(state, gradInput);

    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kH == dH && kW == dW) {
      THError("Not implemented");
//      subgradinput <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
//                                          nInputPlane, nInputRows, nInputCols,
//                                          kH, kW, dH, dW);
    } else {
      THError("Not implemented");
//      subgradinputAtomic <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
//                                                nInputPlane, nInputRows, nInputCols,
//                                                kH, kW, dH, dW);
    }
  }

  return 1;
}

static const struct luaL_Reg clnn_SpatialAveragePooling__ [] = {
  {"SpatialAveragePooling_updateOutput", clnn_SpatialAveragePooling_updateOutput},
  {"SpatialAveragePooling_updateGradInput", clnn_SpatialAveragePooling_updateGradInput},
  {NULL, NULL}
};

static void clnn_SpatialAveragePooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialAveragePooling__, "nn");
  lua_pop(L,1);
}

//#undef CL_MAX_THREADS

