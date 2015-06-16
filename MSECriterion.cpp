#include <iostream>

#include "utils.h"
#include "THClTensor.h"
extern "C" {
  #include "luaT.h"
}

using namespace std;

//#include <thrust/fill.h>
//#include <thrust/functional.h>
//#include <thrust/reduce.h>
//#include <thrust/inner_product.h>

//struct mse_functor
//{
//  mse_functor() {}

//  __host__ __device__ float operator()(const float& x, const float& y) const
//    {
//      float z = x-y;
//      return z*z;
//  }
//};

/*
static int clnn_MSECriterion_updateOutput(lua_State *L)
{
  cout << "MSECriterion_updateOutput start" << endl;
  THClState *state = getCltorchState(L);
  cout << "MSECriterion_updateOutput got state" << endl;
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  cout << "MSECriterion_updateOutput got input" << endl;
  THClTensor *target = (THClTensor*)luaT_checkudata(L, 3, "torch.ClTensor");
  cout << "MSECriterion_updateOutput 2" << endl;
  THAssert(THClTensor_checkGPU(state, 2, input, target));

  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  luaL_argcheck(L, THClTensor_nElement(state, input) == THClTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");

  float sum;

  long size = THClTensor_nElement(state, input);

  input = THClTensor_newContiguous(state, input);
  target = THClTensor_newContiguous(state, target);

  cout << "MSECriterion_updateOutput, before thrust bit" << endl;
  // thrust stuff :-(
  // does something like:
  // sum( per-element-square( per-element-diff( state, input) ) )
  // I guess we can do this in lua???
//  thrust::device_ptr<float> input_data(THClTensor_data(state, input));
//  thrust::device_ptr<float> target_data(THClTensor_data(state, target));
//  sum = thrust::inner_product(input_data, input_data+size, target_data, (float) 0, thrust::plus<float>(), mse_functor());

  if(sizeAverage)
    sum /= size;

  THClTensor_free(state, input);
  THClTensor_free(state, target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  THError("Not implemented");
  return 1;
}
*/

//struct mse_updateGradInput_functor
//{
//  const float norm;

//  mse_updateGradInput_functor(float norm_) : norm(norm_) {}

//  __host__ __device__ float operator()(const float& x, const float& y) const
//    {
//      return norm * (x - y);
//  }
//};

static int clnn_MSECriterion_updateGradInput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *target = (THClTensor*)luaT_checkudata(L, 3, "torch.ClTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THClTensor *gradInput = (THClTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  luaL_argcheck(L, THClTensor_nElement(state, input) == THClTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");
  THAssert(THClTensor_checkGPU(state, 3, input, target, gradInput));

  long size = THClTensor_nElement(state, input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THClTensor_newContiguous(state, input);
  target = THClTensor_newContiguous(state, target);

  THClTensor_resizeAs(state, gradInput, input);

  // no clue what this is ... :-/
   // ok, so this is backprop of gradient for mse, so ...
   // norm is a float
   // and otherwise it's just x - y, which is 
   // looks like this is http://thrust.github.io/doc/group__transformations.html#ga68a3ba7d332887f1332ca3bc04453792
   // so, gradinput = norm * (input - target)
//  thrust::device_ptr<float> input_data(THClTensor_data(state, input));
//  thrust::device_ptr<float> target_data(THClTensor_data(state, target));
//  thrust::device_ptr<float> gradInput_data(THClTensor_data(state, gradInput));

//  thrust::transform(input_data, input_data+size, target_data, gradInput_data, mse_updateGradInput_functor(norm));

  THClTensor_free(state, input);
  THClTensor_free(state, target);
  THError("Not implemented");
  return 1;
}

#define MSECRITERION_THREADS 128


static int clnn_MSECriterion_updateOutput2(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *target = (THClTensor*)luaT_checkudata(L, 3, "torch.ClTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  long size = THClTensor_nElement(state, input);

  input = THClTensor_newContiguous(state, input);
  target = THClTensor_newContiguous(state, target);

  THClStorage *output = THClStorage_newWithSize(state, 1);

  dim3 blocks(1);
  dim3 threads(MSECRITERION_THREADS);

  // kernel launch...
//  clnn_MSECriterion_updateOutput_kernel<<<blocks,threads,
//    0, THClState_getCurrentStream(state)>>>(output->data,
//                                           THClTensor_data(state, input),
//                                           THClTensor_data(state, target),
//                                           1, size,
//                                           sizeAverage);

  lua_pushnumber(L, THClStorage_get(state, output, 0));

  THClTensor_free(state, input);
  THClTensor_free(state, target);
  THClStorage_free(state, output);

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  THError("Not implemented");
  return 1;
}

static int clnn_MSECriterion_updateGradInput2(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *target = (THClTensor*)luaT_checkudata(L, 3, "torch.ClTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THClTensor *gradInput = (THClTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  long size = THClTensor_nElement(state, input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THClTensor_newContiguous(state, input);
  target = THClTensor_newContiguous(state, target);

  THClTensor_resizeAs(state, gradInput, input);

  dim3 blocks(1);
  dim3 threads(MSECRITERION_THREADS);

  // kernel launch...
//  clnn_MSECriterion_updateGradInput_kernel<<<blocks,threads,
//    0, THClState_getCurrentStream(state)>>>(THClTensor_data(state, gradInput),
//                                           THClTensor_data(state, input),
//                                           THClTensor_data(state, target),
//                                           norm,
//                                           1, size);

  THClTensor_free(state, input);
  THClTensor_free(state, target);

  THError("Not implemented");
  return 1;
}


static const struct luaL_Reg clnn_MSECriterion__ [] = {
//  {"MSECriterion_updateOutput", clnn_MSECriterion_updateOutput},
  {"MSECriterion_updateGradInput", clnn_MSECriterion_updateGradInput},
  {"MSECriterion_updateOutput2", clnn_MSECriterion_updateOutput2},
  {"MSECriterion_updateGradInput2", clnn_MSECriterion_updateGradInput2},
  {NULL, NULL}
};

void clnn_MSECriterion_init(lua_State *L)
{
  cout << "clnn_MSECriterion_init" << endl;
  luaT_pushmetatable(L, "torch.ClTensor");
  cout << "clnn_MSECriterion_init pushed metatable" << endl;
  luaT_registeratname(L, clnn_MSECriterion__, "nn");
  cout << "clnn_MSECriterion_init done registeratname" << endl;
  lua_pop(L,1);
}
