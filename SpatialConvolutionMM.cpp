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
#include "conv/ClConvolver.h"
#include "conv/ForwardIm2Col.h"
#include "conv/BackGradIm2Col.h"

#include <iostream>
#include <string>
using namespace std;

// Use 1024 threads per block, which requires cuda sm_2x or above
//const int CL_NUM_THREADS = 1024;

inline int getNumThreads(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

std::string SpatialConvolutionMM_getKernelTemplate();

// CL: number of blocks for threads.
inline int GET_BLOCKS(THClState *state, const int N) {
  return (N + getNumThreads(state) - 1) / getNumThreads(state);
}

void im2col(THClState *state, THClTensor* im, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, THClTensor* col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  std::string uniqueName = "SpatialConvolutionMM::im2col";
  EasyCL *cl = im->storage->cl;
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialConvolutionMM.cl",
      SpatialConvolutionMM_getKernelTemplate(), "im2col_kernel");
  }

  THClKernels k(state, kernel);
  k.in(num_kernels);
  k.in(im);
  k.in(height);
  k.in(width);
  k.in(ksize_h);
  k.in(ksize_w);
  k.in(pad_h);
  k.in(pad_w);
  k.in(stride_h);
  k.in(stride_w);
  k.in(height_col);
  k.in(width_col);
  k.out(col);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));

  // Launch
//  im2col_kernel <<<GET_BLOCKS(num_kernels), CL_NUM_THREADS, 0, stream>>> (
//      num_kernels, data_im, height, width, ksize_h, ksize_w,
//      pad_h, pad_w, stride_h, stride_w,
//      height_col, width_col, data_col
//  );
}

void col2im(THClState *state, THClTensor* col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, THClTensor* im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.

  EasyCL *cl = im->storage->cl;
  std::string uniqueName = "SpatialConvolutionMM::col2im";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialConvolutionMM.cl",
      SpatialConvolutionMM_getKernelTemplate(), "col2im_kernel");
  }

  THClKernels k(state, kernel);
  k.in(num_kernels);
  k.in(col);
  k.in(height);
  k.in(width);
  k.in(channels);

  k.in(patch_h);
  k.in(patch_w);
  k.in(pad_h);
  k.in(pad_w);
  k.in(stride_h);
  k.in(stride_w);

  k.in(height_col);
  k.in(width_col);
  k.out(im);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));

//  col2im_kernel <<<GET_BLOCKS(num_kernels), CL_NUM_THREADS, 0, stream>>> (
//      num_kernels, data_col, height, width, channels,
//      patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
//      height_col, width_col, data_im
//  );
//  THError("Not implemented");
}

static int clnn_SpatialConvolutionMM_updateOutput(lua_State *L) {
//  cout << "clnn_SpatialConvolutionMM_updateOutput(lua_State *L)" << endl;
  THClState *state = getCltorchState(L);
  // Input
  THClTensor *input = (THClTensor*)luaT_checkudata(L, 2, "torch.ClTensor");
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  // Convolver
  lua_getfield(L, 1, "convolver");
  ClConvolver *conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
//  , "nn.ClConvolver")
  if(conv == 0) {
    ClConvolver_new(L);
    lua_pushvalue(L, -1);
    lua_setfield(L, 1, "convolver");
    conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
  }
  if(conv == 0) {
    THError("failed to create convolver object");
  }

  // Params:
  conv->dW = luaT_getfieldcheckint(L, 1, "dW");
  conv->dH = luaT_getfieldcheckint(L, 1, "dH");
  conv->kW = luaT_getfieldcheckint(L, 1, "kW");
  conv->kH = luaT_getfieldcheckint(L, 1, "kH");
  conv->nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  conv->nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  conv->padW = luaT_getfieldcheckint(L, 1, "padW");
  conv->padH = luaT_getfieldcheckint(L, 1, "padH");

  conv->batch = 1;
  if (input->nDimension == 3) {
    luaL_argcheck(L, input->size[0] == conv->nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    conv->batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
  } else {
    luaL_argcheck(L, input->size[1] == conv->nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  conv->inputWidth   = input->size[3];
  conv->inputHeight  = input->size[2];
  conv->outputWidth  = (conv->inputWidth + 2*conv->padW - conv->kW) / conv->dW + 1;
  conv->outputHeight = (conv->inputHeight + 2*conv->padH - conv->kH) / conv->dH + 1;

  if (conv->outputWidth < 1 || conv->outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        conv->nInputPlane,conv->inputHeight,conv->inputWidth,conv->nOutputPlane,conv->outputHeight,conv->outputWidth);

//  conv->inputWidth = inputWidth;
//  conv->inputHeight = inputHeight;
//  conv->outputWidth = outputWidth;
//  conv->outputHeight = outputHeight;

//  conv->dW = dW;
//  conv->dH = dH;
//  conv->kW = kW;
//  conv->kH = kH;
//  conv->nInputPlane = nInputPlane;
//  conv->nOutputPlane = nOutputPlane;
//  conv->padW = padW;
//  conv->padH = padH;

  if(conv->forwarder == 0) {
    conv->forwarder = new ForwardIm2Col(state, input->storage->device, conv);
  }

  THClTensor *weight = (THClTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.ClTensor");
  THClTensor *bias = (THClTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.ClTensor");
  THClTensor *output = (THClTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 4, input, output, weight,
                                 bias));

  // Batch size + input planes
  conv->batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, output, conv->batchSize, conv->nOutputPlane, conv->outputHeight, conv->outputWidth);

  conv->forwarder->forward(state, input, weight, bias, output);

  // return output
  return 1;
}

static int clnn_SpatialConvolutionMM_updateGradInput(lua_State *L) {
  THClState *state = getCltorchState(L);
  // Inputs
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");

  lua_getfield(L, 1, "convolver");
  ClConvolver *conv = (ClConvolver*)luaT_toudata(L, -1, "nn.ClConvolver");
  if(conv == 0) {
    THError("clconvolver object not found");
  }

  // Params
//  int dW = luaT_getfieldcheckint(L, 1, "dW");
//  int dH = luaT_getfieldcheckint(L, 1, "dH");
//  int kW = luaT_getfieldcheckint(L, 1, "kW");
//  int kH = luaT_getfieldcheckint(L, 1, "kH");
//  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
//  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
//  int padW = luaT_getfieldcheckint(L, 1, "padW");
//  int padH = luaT_getfieldcheckint(L, 1, "padH");

  THClTensor *weight = (THClTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.ClTensor");
//  THClTensor *gradColumns = (THClTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.ClTensor");
  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 4, input, gradOutput, weight,
                                 gradInput));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  conv->batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    conv->batch = 0;
    THClTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THClTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  if(conv->backGradder == 0) {
    conv->backGradder = new BackGradIm2Col(state, input->storage->device, conv);
  }

//  long inputWidth   = input->size[3];
//  long inputHeight  = input->size[2];
//  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
//  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
//  long batchSize = input->size[0];

  // Resize output
  THClTensor_resize4d(state, gradInput, conv->batchSize, conv->nInputPlane, conv->inputHeight, conv->inputWidth);

  conv->backGradder->updateGradInput(state, input, gradOutput, weight, gradInput);

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
//  cout << "registering spatialconvolutionmm" << endl;
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialConvolutionMM__, "nn");
  lua_pop(L,1);
}

#undef CL_KERNEL_LOOP

std::string SpatialConvolutionMM_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "SpatialConvolutionMM.cl" )
  // ]]]
  // generated using cog, from SpatialConvolutionMM.cl:
  const char * kernelSource =  
  "// from SpatialConvolutionMM.cu:\n" 
  "\n" 
  "// CL: grid stride looping\n" 
  "#define CL_KERNEL_LOOP(i, n)                        \\\n" 
  "  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\\n" 
  "      i < (n);                                       \\\n" 
  "      i += get_local_size(0) * get_num_groups(0))\n" 
  "\n" 
  "// Kernel for fast unfold+copy\n" 
  "// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)\n" 
  "kernel void im2col_kernel(const int n, const global float* im_data, int im_offset,\n" 
  "    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,\n" 
  "    const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,\n" 
  "    global float* col_data, int col_offset) {\n" 
  "  global const float *data_im = im_data + im_offset;\n" 
  "  global float *data_col = col_data + col_offset;\n" 
  "\n" 
  "  CL_KERNEL_LOOP(index, n) {\n" 
  "    int w_out = index % width_col;\n" 
  "    index /= width_col;\n" 
  "    int h_out = index % height_col;\n" 
  "    int channel_in = index / height_col;\n" 
  "    int channel_out = channel_in * ksize_h * ksize_w;\n" 
  "    int h_in = h_out * stride_h - pad_h;\n" 
  "    int w_in = w_out * stride_w - pad_w;\n" 
  "    data_col += (channel_out * height_col + h_out) * width_col + w_out;\n" 
  "    data_im += (channel_in * height + h_in) * width + w_in;\n" 
  "    for (int i = 0; i < ksize_h; ++i) {\n" 
  "      for (int j = 0; j < ksize_w; ++j) {\n" 
  "        int h = h_in + i;\n" 
  "        int w = w_in + j;\n" 
  "        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?\n" 
  "          data_im[i * width + j] : 0;\n" 
  "        data_col += height_col * width_col;\n" 
  "      }\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "kernel void col2im_kernel(const int n, global const float* col_data, int col_offset,\n" 
  "    const int height, const int width, const int channels, const int patch_h, const int patch_w,\n" 
  "    const int pad_h, const int pad_w, const int stride_h, const int stride_w,\n" 
  "    const int height_col, const int width_col,\n" 
  "    global float* im_data, int im_offset) {\n" 
  "  global float *data_im = im_data + im_offset;\n" 
  "  global const float *data_col = col_data + col_offset;\n" 
  "\n" 
  "  CL_KERNEL_LOOP(index, n) {\n" 
  "    float val = 0;\n" 
  "    int w = index % width + pad_w;\n" 
  "    int h = (index / width) % height + pad_h;\n" 
  "    int c = index / (width * height);\n" 
  "    // compute the start and end of the output\n" 
  "    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;\n" 
  "    int w_col_end = min(w / stride_w + 1, width_col);\n" 
  "    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;\n" 
  "    int h_col_end = min(h / stride_h + 1, height_col);\n" 
  "    /*\n" 
  "       for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n" 
  "       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n" 
  "    // the col location: [c * width * height + h_out, w_out]\n" 
  "    int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize + (w - w_col * stride_w);\n" 
  "    val += data_col[(c_col * height_col + h_col) * width_col + w_col];\n" 
  "    }\n" 
  "    }\n" 
  "     */\n" 
  "    // equivalent implementation\n" 
  "    int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;\n" 
  "    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;\n" 
  "    int coeff_w_col = (1 - stride_w * height_col * width_col);\n" 
  "    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n" 
  "      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n" 
  "        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];\n" 
  "      }\n" 
  "    }\n" 
  "    data_im[index] = val;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

