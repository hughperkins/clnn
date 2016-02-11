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

TH_API void THNN_ClELU_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output,
          float alpha);
TH_API void THNN_ClELU_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput,
          THClTensor *output,
          float alpha);

TH_API void THNN_ClTanh_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output);
TH_API void THNN_ClTanh_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput,
          THClTensor *output);

TH_API void THNN_ClSpatialConvolutionMM_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output,
          THClTensor *weight,
          THClTensor *bias,
          THClTensor *columns,
          THClTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_ClSpatialConvolutionMM_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput,
          THClTensor *weight,
          THClTensor *bias,
          THClTensor *columns,
          THClTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_ClSpatialConvolutionMM_accGradParameters(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradWeight,
          THClTensor *gradBias,
          THClTensor *columns,
          THClTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          float scale);
TH_API void THNN_ClSpatialAveragePooling_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);
TH_API void THNN_ClSpatialAveragePooling_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);

TH_API void THNN_ClSpatialMaxPooling_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output,
          THClTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
TH_API void THNN_ClSpatialMaxPooling_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput,
          THClTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
TH_API void THNN_ClSoftMax_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output);
TH_API void THNN_ClSoftMax_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput,
          THClTensor *output);

TH_API void THNN_ClSpatialUpSamplingNearest_updateOutput(
        THClState *state,
        THClTensor *input,
        THClTensor *output,
        int scale_factor);
TH_API void THNN_ClSpatialUpSamplingNearest_updateGradInput(
        THClState *state,
        THClTensor *input,
        THClTensor *gradOutput,
        THClTensor *gradInput,
        int scale_factor);
