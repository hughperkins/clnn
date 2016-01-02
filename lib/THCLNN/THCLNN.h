// from THCUNN.h:

#include <THCl/THCl.h>
#include "THClApply.h"

TH_API void THNN_ClAbs_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output);
TH_API void THNN_ClAbs_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput);

//TH_API void THNN_ClAbsCriterion_updateOutput(
//          THClState *state,
//          THClTensor *input,
//          THClTensor *target,
//          float *output,
//          bool sizeAverage);
//TH_API void THNN_ClAbsCriterion_updateGradInput(
//          THClState *state,
//          THClTensor *input,
//          THClTensor *target,
//          THClTensor *gradInput,
//          bool sizeAverage);

