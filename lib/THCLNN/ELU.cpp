// from ELU.cu:

#include "utils.h"
#include "luaT.h"
#include "THClApply.h"
#include "THCLNN.h"

#include <iostream>
using namespace std;

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

void THNN_ClELU_updateOutput(THClState *state, THClTensor *input, THClTensor *output, float alpha)
{
  THAssert(THClTensor_checkGPU(state, 2, input, output));
  THClTensor_resizeAs(state, output, input);
  ELUupdateOutput_functor func(alpha);
  THClTensor_pointwiseApply2(state, output, input, &func);
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

void THNN_ClELU_updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, 
  THClTensor *gradInput, THClTensor *output, float alpha)
{
  THAssert(THClTensor_checkGPU(state, 3, output, gradOutput, gradInput));
  THClTensor_resizeAs(state, gradInput, output);
  ELUupdateGradInput_functor func(alpha);
  THClTensor_pointwiseApply3(state, gradInput, output, gradOutput, &func);
}
