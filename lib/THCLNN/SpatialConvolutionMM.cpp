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
#include "THCLNN.h"
#include "im2col.h"

#include <iostream>
#include <string>
using namespace std;

void THNN_ClSpatialConvolutionMM_updateOutput(THClState *state, THClTensor *input, THClTensor *output, THClTensor *weight, THClTensor *bias, THClTensor *columns, THClTensor *ones, int kW, int kH, int dW, int dH, int padW, int padH) {
  THAssert(THClTensor_checkGPU(state, 6, input, output, weight,
                                 bias, columns, ones));
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  THArgCheck(weight->nDimension == 2, 4, "weight tensor must be 2D (nOutputPlane,nInputPlane*kH*kW)");
  THArgCheck(weight->size[0] == bias->size[0], 4, "nOutputPlane mismatch in weight and bias");
  THArgCheck(kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params:
  int nInputPlane = weight->size[1]/(kH*kW);
  int nOutputPlane = weight->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    THArgCheck(input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
  } else {
    THArgCheck(input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THClTensor_resize2d(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THClTensor_resize2d(state, ones, outputHeight, outputWidth);
    THClTensor_fill(state, ones, 1);
  }

  // Helpers
  THClTensor *input_n = THClTensor_newv2(state, input->storage->device);
  THClTensor *output_n = THClTensor_newv2(state, input->storage->device);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THClTensor_select(state, input_n, input, 0, elt);
    THClTensor_select(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
//    cout << "THCLBlas_Gemm('t', 'n', " << n_ << ", " << m_ << " ," << k_ << ", 1, ones, " << k_ << ", bias, " << k_
//       << ", 0, output_n, " << n_ << ");" << endl;
    THClBlas_gemm(
        state,
        'n', 'n',
        n_, m_, k_,
        1,
        ones, k_,
        bias, k_,
        0,
        output_n, n_
    );

    // Extract columns:
    im2col(
      state,
//      THClState_getCurrentStream(state),
      input_n,
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      columns
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
//    cout << "THCLBlas_Gemm('n', 'n', " << n << ", " << m << " ," << k << ", 1, columns, " << n << ", weight, " << k
//       << ", 1, output_n, " << n << ");" << endl;
    THClBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        columns, n,
        weight, k,
        1,
        output_n, n
    );
  }

  // Free
  THClTensor_free(state, input_n);
  THClTensor_free(state, output_n);

  // Resize output
  if (batch == 0) {
    THClTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
    THClTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }
}

void THNN_ClSpatialConvolutionMM_updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradInput, THClTensor *weight, THClTensor *bias, THClTensor *gradColumns, THClTensor *ones, int kW, int kH, int dW, int dH, int padW, int padH) {

  THAssert(THClTensor_checkGPU(state, 5, input, gradOutput, weight,
                                 gradColumns, gradInput));
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  THArgCheck(weight->nDimension == 2, 4, "weight tensor must be 2D (nOutputPlane,nInputPlane*kH*kW)");
  THArgCheck(weight->size[0] == bias->size[0], 4, "nOutputPlane mismatch in weight and bias");
  THArgCheck(kW > 0 && kH > 0, 9, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 11, "stride should be greater than zero");

  // Params
  int nInputPlane = weight->size[1]/(kW*kH);
  int nOutputPlane = weight->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THClTensor_resize2d(state, gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THClTensor *gradInput_n = THClTensor_newv2(state, input->storage->device);
  THClTensor *gradOutput_n = THClTensor_newv2(state, input->storage->device);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THClTensor_select(state, gradInput_n, gradInput, 0, elt);
    THClTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = nInputPlane*kW*kH;
    long n = gradColumns->size[1];
    long k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        'n', 't',
        n, m, k,
        1,
        gradOutput_n, n,
        weight, m,
        0,
        gradColumns, n
    );

    // Unpack columns back into input:
    col2im(
      state,
//      THClState_getCurrentStream(state),
      gradColumns,
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      gradInput_n
    );
  }

  // Free
  THClTensor_free(state, gradInput_n);
  THClTensor_free(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THClTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THClTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THClTensor_resize3d(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }
}

void THNN_ClSpatialConvolutionMM_accGradParameters(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradWeight, THClTensor *gradBias, THClTensor *columns, THClTensor *ones, int kW, int kH, int dW, int dH, int padW, int padH, float scale) {

  THAssert(THClTensor_checkGPU(state, 6, input, gradOutput, gradWeight,
                                 gradBias, columns, ones));
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  THArgCheck(gradWeight->nDimension == 2, 4, "gradWeight tensor must be 2D (nOutputPlane,nInputPlane*kH*kW)");
  THArgCheck(gradWeight->size[0] == gradBias->size[0], 4, "nOutputPlane mismatch in gradWeight and gradBias");
  THArgCheck(kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params
  int nInputPlane = gradWeight->size[1]/(kW*kH);
  int nOutputPlane = gradWeight->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THClTensor_resize2d(state, ones, outputHeight, outputWidth);
    THClTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THClTensor_resize2d(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THClTensor *input_n = THClTensor_newv2(state, input->storage->device);
  THClTensor *gradOutput_n = THClTensor_newv2(state, input->storage->device);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THClTensor_select(state, input_n, input, 0, elt);
    THClTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
      state,
//      THClState_getCurrentStream(state),
      input_n,
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      columns
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = nInputPlane*kW*kH;
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        't', 'n',
        n, m, k,
        scale,
        columns, k,
        gradOutput_n, k,
        1,
        gradWeight, n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    THClBlas_gemv(
        state,
        't',
        k_, m_,
        scale,
        gradOutput_n, k_,
        ones, 1,
        1,
        gradBias, 1
    );
  }

  // Free
  THClTensor_free(state, input_n);
  THClTensor_free(state, gradOutput_n);

  // Resize
  if (batch == 0) {
    THClTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THClTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }
}

