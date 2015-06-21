// from SpatialConvolutionMM.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include <iostream>
using namespace std;

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CL_NUM_THREADS = 1024;

// CL: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CL_NUM_THREADS - 1) / CL_NUM_THREADS;
}

void im2col(THClState *state, THClTensor* im, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, THClTensor* col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
//  im2col_kernel <<<GET_BLOCKS(num_kernels), CL_NUM_THREADS, 0, stream>>> (
//      num_kernels, data_im, height, width, ksize_h, ksize_w,
//      pad_h, pad_w, stride_h, stride_w,
//      height_col, width_col, data_col
//  );
  THError("Not implemented");
}

void col2im(THClState *state, THClTensor* col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, THClTensor* im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
//  col2im_kernel <<<GET_BLOCKS(num_kernels), CL_NUM_THREADS, 0, stream>>> (
//      num_kernels, data_col, height, width, channels,
//      patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
//      height_col, width_col, data_im
//  );
  THError("Not implemented");
}

static int clnn_SpatialConvolutionMM_updateOutput(lua_State *L) {
  cout << "clnn_SpatialConvolutionMM_updateOutput(lua_State *L)" << endl;
  THClState *state = getCltorchState(L);
  // Input
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");

//  cout << "dW=" << dW << " dH=" << dH << " kW=" << kW << " kH=" << kH 
//    << " nInputPlane=" << nInputPlane << " nOutputPlane=" << nOutputPlane
//   << " padding=" << padding << endl;

  THClTensor *weight = (THClTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.ClTensor");
  THClTensor *bias = (THClTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.ClTensor");
  THClTensor *columns = (THClTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.ClTensor");
  THClTensor *ones = (THClTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.ClTensor");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 6, input, output, weight,
                                 bias, columns, ones));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    luaL_argcheck(L, input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
  } else {
    luaL_argcheck(L, input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
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
  THClTensor *input_n = THClTensor_new(state);
  THClTensor *output_n = THClTensor_new(state);

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
    THClBlas_gemm(
        state,
        't', 'n',
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
    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
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

  // return output
  return 1;
}

static int clnn_SpatialConvolutionMM_updateGradInput(lua_State *L) {
  THClState *state = getCltorchState(L);
  // Inputs
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");

  THClTensor *weight = (THClTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.ClTensor");
  THClTensor *gradColumns = (THClTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.ClTensor");
  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 5, input, gradOutput, weight,
                                 gradColumns, gradInput));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

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
  THClTensor *input_n = THClTensor_new(state);
  THClTensor *gradInput_n = THClTensor_new(state);
  THClTensor *gradOutput_n = THClTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THClTensor_select(state, input_n, input, 0, elt);
    THClTensor_select(state, gradInput_n, gradInput, 0, elt);
    THClTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = weight->size[1];
    long n = gradColumns->size[1];
    long k = weight->size[0];

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
  THClTensor_free(state, input_n);
  THClTensor_free(state, gradInput_n);
  THClTensor_free(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THClTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THClTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THClTensor_resize3d(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }

  // Return gradInput
  return 1;
}

static int clnn_SpatialConvolutionMM_accGradParameters(lua_State *L) {
  THClState *state = getCltorchState(L);
  // Inputs
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  float scale = luaL_optnumber(L, 4, 1);

  THClTensor *gradWeight = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.ClTensor");
  THClTensor *gradBias = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.ClTensor");
  THClTensor *columns = (THClTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.ClTensor");
  THClTensor *ones = (THClTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 6, input, gradOutput, gradWeight,
                                 gradBias, columns, ones));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

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
  THClTensor *input_n = THClTensor_new(state);
  THClTensor *gradOutput_n = THClTensor_new(state);

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
    long m = gradWeight->size[0];
    long n = gradWeight->size[1];
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

  // Return nothing
  return 0;
}

static const struct luaL_Reg clnn_SpatialConvolutionMM__ [] = {
  {"SpatialConvolutionMM_updateOutput", clnn_SpatialConvolutionMM_updateOutput},
  {"SpatialConvolutionMM_updateGradInput", clnn_SpatialConvolutionMM_updateGradInput},
  {"SpatialConvolutionMM_accGradParameters", clnn_SpatialConvolutionMM_accGradParameters},
  {NULL, NULL}
};

void clnn_SpatialConvolutionMM_init(lua_State *L)
{
  cout << "registering spatialconvolutionmm" << endl;
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialConvolutionMM__, "nn");
  lua_pop(L,1);
}

#undef CL_KERNEL_LOOP

