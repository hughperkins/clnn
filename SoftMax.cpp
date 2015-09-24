// from SoftMax.cu:

#include "utils.h"

//#define MINUS_LOG_THRESHOLD -18.42
//#define SOFTMAX_THREADS 128


static int clnn_SoftMax_updateOutput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  THAssert(THClTensor_checkGPU(state, 2, input, output));

  input = THClTensor_newContiguous(state, input);
  THClTensor_resizeAs(state, output, input);
  long batchSize, dim, stride;

  if(input->nDimension == 1)
  {
    batchSize = 1;
    dim = input->size[0];
    stride = 1;
  }
  else if(input->nDimension == 2)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride = 1;
  }
  else if(input->nDimension == 3)
  {
    batchSize = 1;
    dim = input->size[0];
    stride = input->size[1]*input->size[2];
  }
  else if(input->nDimension == 4)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride = input->size[2]*input->size[3];
  }
  else
    THError("1D, 2D, 3D or 4D tensor expected");

  dim3 blocks(batchSize, stride);
  dim3 threads(SOFTMAX_THREADS);
  clnn_SoftMax_updateOutput_kernel<<<blocks,threads,
    0, THClState_getCurrentStream(state)>>>(THClTensor_data(state, output),
                                           THClTensor_data(state, input),
                                           batchSize, dim, stride);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THClTensor_free(state, input);
  return 1;
}

static int clnn_SoftMax_updateGradInput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *gradOutput = (THClTensor*)luaT_checkudata(L, 3, "torch.ClTensor");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  THClTensor *gradInput = (THClTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  THAssert(THClTensor_checkGPU(state, 3, output, gradOutput, gradInput));

  output = THClTensor_newContiguous(state, output);
  gradOutput = THClTensor_newContiguous(state, gradOutput);

  THClTensor_resizeAs(state, gradInput, output);
  long batchSize, dim, stride;

  if(gradInput->nDimension == 1)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride = 1;
  }
  else if(gradInput->nDimension == 2)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride = 1;
  }
  else if(gradInput->nDimension == 3)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride = gradInput->size[1]*gradInput->size[2];
  }
  else if(gradInput->nDimension == 4)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride = gradInput->size[2]*gradInput->size[3];
  }
  else
    THError("1D, 2D, 3D or 4D tensor expected");

  dim3 blocks(batchSize, stride);
  dim3 threads(SOFTMAX_THREADS);
  clnn_SoftMax_updateGradInput_kernel<<<blocks,threads,
    0, THClState_getCurrentStream(state)>>>(THClTensor_data(state, gradInput),
                                           THClTensor_data(state, output),
                                           THClTensor_data(state, gradOutput),
                                           batchSize, dim, stride);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THClTensor_free(state, gradOutput);
  THClTensor_free(state, output);
  return 1;
}

static const struct luaL_Reg clnn_SoftMax__ [] = {
  {"SoftMax_updateOutput", clnn_SoftMax_updateOutput},
  {"SoftMax_updateGradInput", clnn_SoftMax_updateGradInput},
  {NULL, NULL}
};

void clnn_SoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SoftMax__, "nn");
  lua_pop(L,1);
}

