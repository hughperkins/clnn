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
#include "THClDebug.h"

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

  long inW   = input->size[3];
  long inH  = input->size[2];
  long outW  = (inW + 2*padW - kW) / dW + 1;
  long outH = (inH + 2*padH - kH) / dH + 1;

  if (outW < 1 || outH < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inH,inW,nOutputPlane,outH,outW);

  // following "OpenCL caffe: Accelerating and enabling a cross platform machine" by Junli Gu et al, we're
  // going to group in thisGroupSize groups, up to 16
  // this means concretely:
  // - no change to weights.  I think.  yay :-)
  // output will have 16 times more rows
  // columns will have 16 times more columns, ie thisGroupSize * outH * outW
  // so.... easiest thing will be to run im2col thisGroupSize times, once for each image
  int desiredGroupSize = 16;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, output, batchSize, nOutputPlane, outH, outW);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 1 || ones->size[0] < outH*outW*desiredGroupSize) {
    // Resize plane and fill with ones...
    THClTensor_resize1d(state, ones, desiredGroupSize * outH * outW);
    THClTensor_fill(state, ones, 1);
  }

  // Helpers
  THClTensor *input_n = THClTensor_newv2(state, input->storage->device);
  THClTensor *output_n = THClTensor_newv2(state, input->storage->device);
  THClTensor *outputBatch = THClTensor_newWithSize2d(state, input->storage->device,
    nOutputPlane, outW * outH * desiredGroupSize); // this should be persisted somehow

  int numGroups = (batchSize + desiredGroupSize - 1) / desiredGroupSize;
  for(int g=0; g < numGroups; g++) {
    int eltStart = g * desiredGroupSize;
    int eltEnd = eltStart + desiredGroupSize; // we will exclude this value in iteration
    int thisGroupSize = eltEnd - eltStart;
    if(g == numGroups - 1) {
      thisGroupSize = batchSize - eltStart;
      eltEnd = eltStart + thisGroupSize;
    }

    THClTensor_resize1d(state, ones, thisGroupSize * outH * outW);
    // no need to fill with 1s, since thisGroupSize <= desiredGroupSize, and we already filled that many
    // 1s
    THClTensor_resize2d(state, outputBatch, nOutputPlane, thisGroupSize * outH * outW);

    // Resize temporary columns
    THClTensor_resize2d(state, columns, nInputPlane*kW*kH, thisGroupSize * outH*outW);

    // weights should already be ok, but we need to run im2col, into columns
    for(int elt = eltStart; elt < eltEnd; elt++) {
      // Extract columns:
      THClTensor_select(state, input_n, input, 0, elt);
      im2col_batched(
        state,
        input_n,
        nInputPlane, inW, inH, kW, kH, dW, dH, padW, padH,
        thisGroupSize, elt - eltStart,
        columns
      );
    }

    // Do Bias first:
    THClBlas_gemm2(
        state,
         'r',
        't', 't',
        nOutputPlane, outW * outH * thisGroupSize, 1,
        1,
        bias, nOutputPlane,
        ones, 1,
        0,
        outputBatch, outW * outH * thisGroupSize
    );

    THClBlas_gemm2(
        state,
        'r',
        'n', 'n',
        nOutputPlane, outW * outH * thisGroupSize, nInputPlane*kH*kW,
        1,
        weight, nInputPlane*kH*kW,
        columns, outW * outH * thisGroupSize,
        1,
        outputBatch, outW * outH * thisGroupSize
    );

    // copy from outputBatch to output_n
    THClTensor_narrow(state, output_n, output, 0, eltStart, thisGroupSize);
    for(int image=0; image < thisGroupSize; image++) {
      THClTensor *src = THClTensor_newWithStorage2d(state, input->storage->device,
        THClTensor_storage(state, outputBatch), THClTensor_storageOffset(state, outputBatch) +
          image * outW * outH,
        nOutputPlane, outW * outH * thisGroupSize,
        outW * outH, 1);
      THClTensor *dest = THClTensor_newNarrow(state, output_n,
        0, image, 1);
      THClTensor_copyCl(state, dest, src);
      THClTensor_free(state, src);
      THClTensor_free(state, dest);
    }

//weight: outPlanes, inPlanes * kW * kH
//columns: inPlanes * kW * kH, outW * outH
//output: outPlanes, outW * outH
  }

  // Free
  THClTensor_free(state, input_n);
  THClTensor_free(state, outputBatch);
  THClTensor_free(state, output_n);

  // Resize output
  if (batch == 0) {
    THClTensor_resize3d(state, output, nOutputPlane, outH, outW);
    THClTensor_resize3d(state, input, nInputPlane, inH, inW);
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

  long inW   = input->size[3];
  long inH  = input->size[2];
  long outW  = (inW + 2*padW - kW) / dW + 1;
  long outH = (inH + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, gradInput, batchSize, nInputPlane, inH, inW);

  // Resize temporary columns
  THClTensor_resize2d(state, gradColumns, nInputPlane*kW*kH, outH*outW);

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
//    long m = nInputPlane*kW*kH;
//    long n = outH*outW;
//    long k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm2(
        state,
        'r',
        't', 'n',
        nInputPlane*kW*kH, outH*outW, nOutputPlane,
        1,
        weight, nInputPlane*kW*kH,
        gradOutput_n, outH*outW,
        0,
        gradColumns, outH*outW
    );
//    cout << "gradColumns" << endl;
//    THClDebug_printTensor(state, gradColumns);

//weight: nOutputPlane, nInputPlane * kW * kH
// weight:t() nInputPlane * kW * kH, nOutputPlane
//output: nOutputPlane, outW * outH
//columns: nInputPlane * kW * kH, outW * outH

    // Unpack columns back into input:
    col2im(
      state,
//      THClState_getCurrentStream(state),
      gradColumns,
      nInputPlane,
      inW, inH, kW, kH, dW, dH, padW, padH,
      gradInput_n
    );
  }

  // Free
  THClTensor_free(state, gradInput_n);
  THClTensor_free(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THClTensor_resize3d(state, gradOutput, nOutputPlane, outH, outW);
    THClTensor_resize3d(state, input, nInputPlane, inH, inW);
    THClTensor_resize3d(state, gradInput, nInputPlane, inH, inW);
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

  long inW   = input->size[3];
  long inH  = input->size[2];
  long outW  = (inW + 2*padW - kW) / dW + 1;
  long outH = (inH + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outH*outW) {
    // Resize plane and fill with ones...
    THClTensor_resize2d(state, ones, outH, outW);
    THClTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THClTensor_resize2d(state, columns, nInputPlane*kW*kH, outH*outW);

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
      nInputPlane, inW, inH, kW, kH, dW, dH, padW, padH,
      columns
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = nInputPlane*kW*kH;
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm2(
        state,
        'c',
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
    long k_ = outH * outW;

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
    THClTensor_resize3d(state, gradOutput, nOutputPlane, outH, outW);
    THClTensor_resize3d(state, input, nInputPlane, inH, inW);
  }
}

