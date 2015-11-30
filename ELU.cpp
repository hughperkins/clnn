// from ELU.cu:

#include "utils.h"
#include "luaT.h"
#include "THClApply.h"

class ELUupdateOutput_functor : public HasOperator1, public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar(int index) const { return alpha_; }

  const float alpha_;

  ELUupdateOutput_functor(float alpha) : alpha_(alpha) {}

  std::string operator1() const {
    return "*out = *out <= 0 ? (exp(*out)-1) * val1 : *out";
  }
  std::string operator2() const {
    return "*out = *in1 <= 0 ? (exp(*in1)-1) * val1 : *in1";
  }
};

static int clnn_ELU_updateOutput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  THAssert(THClTensor_checkGPU(state, 2, input, output));
  THClTensor_resizeAs(state, output, input);
  ELUupdateOutput_functor func(alpha);
  THClTensor_pointwiseApply2(state, output, input, &func);
  return 1;
}

class ELUupdateGradInput_functor : public HasOperator2, public HasOperator3, public HasScalars
{
public:
  int getNumScalars() const { return 1; }
  float getScalar(int index) const { return alpha_; }

  const float alpha_;

  ELUupdateGradInput_functor(float alpha) : alpha_(alpha) {}

  std::string operator2() const {
    return "*out = (*out) <= 0 ? (*in1 * (*out + val1)) : (*in1)";
  }
  std::string operator3() const {
    return "*out = (*in1) <= 0 ? (*in2 * (*in1 + val1)) : (*in2)";
  }
};

static int clnn_ELU_updateGradInput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *gradOutput = (THClTensor*)luaT_checkudata(L, 3, "torch.ClTensor");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  THClTensor *gradInput = (THClTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  THAssert(THClTensor_checkGPU(state, 3, output, gradOutput, gradInput));
  THClTensor_resizeAs(state, gradInput, output);
  ELUupdateGradInput_functor func(alpha);
  THClTensor_pointwiseApply3(state, gradInput, output, gradOutput, &func);
  return 1;
}

static const struct luaL_Reg clnn_ELU__ [] = {
  {"ELU_updateOutput", clnn_ELU_updateOutput},
  {"ELU_updateGradInput", clnn_ELU_updateGradInput},
  {NULL, NULL}
};

void clnn_ELU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_ELU__, "nn");
  lua_pop(L,1);
}

