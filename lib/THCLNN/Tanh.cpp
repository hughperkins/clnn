// from Tanh.cu:

#include "THCLNN.h"

void THNN_ClTanh_updateOutput(THClState *state, THClTensor *input, THClTensor *output)
{
  THAssert(THClTensor_checkGPU(state, 2, input, output));
  THClTensor_resizeAs(state, output, input);
  THClTensor_tanh(state, output, input);
}

void THNN_ClTanh_updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradInput, THClTensor *output)
{
  THAssert(THClTensor_checkGPU(state, 3, output, gradOutput, gradInput));
  THClTensor_resizeAs(state, gradInput, output);
  THClTensor_map2(state, gradInput, gradOutput, output, "*out = *in1 * (1 - *in2 * *in2)");
}

