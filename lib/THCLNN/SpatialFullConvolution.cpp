// from SpatialFullConvolution.cu:

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

void THNN_ClSpatialFullConvolution_updateOutput(
  THClState *state,
  THClTensor *input,
  THClTensor *output,
  THClTensor *weight,
  THClTensor *bias,
  THClTensor *columns,
  THClTensor *ones,
  int nInputPlane,
  int nOutputPlane,
  int kW, int kH,
  int dW, int dH,
  int padW, int padH,
  int adjW, int adjH) {
  THAssert(THClTensor_checkGPU(state, 6, input, output, weight,
                                 bias, columns, ones));
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  //int nInputPlane = THCudaTensor_size(state, weight, 0);
  //int nOutputPlane = THCudaTensor_size(state, weight, 1);
  
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
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THClTensor_resize2d(state, columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

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

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = weight->size[1] * weight->size[2] * weight->size[3];
    long n = columns->size[1];
    long k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        'n', 't',
        n, m, k,
        1,
        input_n, n,
        weight, m,
        0,
        columns, n
    );

    // Unpack columns back into input:
    col2im(
     state,
      //THClState_getCurrentStream(state),
      columns,
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      output_n
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        't', 'n',
        n_, m_, k_,
        1,
        ones, k_,
        bias, k_,
        1,
        output_n, n_
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

  // return output
  //return 1;
}

void THNN_ClSpatialFullConvolution_updateGradInput(
  THClState *state,
  THClTensor *input,
  THClTensor *gradOutput,
  THClTensor *gradInput,
  THClTensor *weight,
  THClTensor *gradColumns,
  int nInputPlane,  
  int nOutputPlane,
  int kW, int kH,
  int dW, int dH,
  int padW, int padH,
  int adjW, int adjH) {
  
  //int nInputPlane = THCudaTensor_size(state, weight, 0);
  //int nOutputPlane = THCudaTensor_size(state, weight, 1);

  THAssert(THClTensor_checkGPU(state, 5, input, gradOutput, weight,
                                 gradColumns, gradInput));
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");


  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THClTensor_resize2d(state, gradColumns, nOutputPlane*kW*kH, inputHeight*inputWidth);

  // Helpers
  THClTensor *gradInput_n = THClTensor_newv2(state, input->storage->device);
  THClTensor *gradOutput_n = THClTensor_newv2(state, input->storage->device);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THClTensor_select(state, gradInput_n, gradInput, 0, elt);
    THClTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
      state,
      //THClState_getCurrentStream(state),
      gradOutput_n,
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      gradColumns
    );


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = gradColumns->size[1];
    long k = weight->size[1] * weight->size[2] * weight->size[3];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        gradColumns, n,
        weight, k,
        0,
        gradInput_n, n
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

  // Return gradInput
  //return 1;
}


void THNN_ClSpatialFullConvolution_accGradParameters(
  THClState *state,
  THClTensor *input,
  THClTensor *gradOutput,
  THClTensor *gradWeight,
  THClTensor *gradBias,
  THClTensor *columns,
  THClTensor *ones,
  int nInputPlane,  
  int nOutputPlane,
  int kW, int kH,
  int dW, int dH,
  int padW, int padH,
  int adjW, int adjH,
  float scale) {

  THAssert(THClTensor_checkGPU(state, 6, input, gradOutput, gradWeight,
                                 gradBias, columns, ones));
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");


  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THClTensor_resize2d(state, ones, outputHeight, outputWidth);
    THClTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THClTensor_resize2d(state, columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

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
      //THClState_getCurrentStream(state),
      gradOutput_n,
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      columns
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long n = columns->size[0];   // nOutputPlane * kh * kw
    long m = input_n->size[0];   // nInputPlane
    long k = columns->size[1];   // inputHeight * inputWidth

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        't', 'n',
        n, m, k,
        scale,
        columns, k,
        input_n, k,
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

  // Return nothing
  //return 0;
}
/* there's no reference for this is SpatialConvolutionMM 
const struct luaL_Reg clnn_SpatialFullConvolution__ [] = {
  {"SpatialFullConvolution_updateOutput", clnn_SpatialFullConvolution_updateOutput},
  {"SpatialFullConvolution_updateGradInput", clnn_SpatialFullConvolution_updateGradInput},
  {"SpatialFullConvolution_accGradParameters", clnn_SpatialFullConvolution_accGradParameters},
  {NULL, NULL}
};

void clnn_SpatialFullConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialFullConvolution__, "nn");
  lua_pop(L,1);
}
*/

/* Hmm, nope, need to call from im2col perhaps?
std::string THNN_ClSpatialFullConvolution_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "lib/THCLNN/SpatialFullConvolution.cl" )
  // ]]]
  // generated using cog, from lib/THCLNN/SpatialFullConvolution.cl:
  const char * kernelSource =  
  "// from SpatialFullConvolution.cu:\n"
  "\n"
  "";
  // [[[end]]]
  return kernelSource;
}
*/
