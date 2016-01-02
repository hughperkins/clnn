// from Abs.cu:

#include "THCLNN.h"

#include <string>
using namespace std;

class ClnnAbs_forward : public HasOperator2 {
public:
  string operator2() const {
    return "*out = abs(*in1)";
  }
};

class ClnnAbs_backward : public HasOperator3 {
public:
  string operator3() const {
    return "*out = *in1 < 0 ? - *in2 : *in2";
  }
};

void THNN_ClAbs_updateOutput(THClState *state, THClTensor *input, THClTensor *output)
{
  THAssert(THClTensor_checkGPU(state, 2, input, output));
  THClTensor_resizeAs(state, output, input);
  ClnnAbs_forward op;
  THClTensor_pointwiseApply2(state, output, input, &op);
}

void THNN_ClAbs_updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradInput)
{
  THAssert(THClTensor_checkGPU(state, 3, input, gradOutput, gradInput));
  THClTensor_resizeAs(state, gradInput, input);
  ClnnAbs_backward op;
  THClTensor_pointwiseApply3(state, gradInput, input, gradOutput, &op);
}

