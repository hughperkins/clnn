// from SpatialMaxPooling.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "EasyCL.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"
#include "common.h"

#include <iostream>
#include <string>
using namespace std;

// groupsize: kW x kH
// stride: dW, dH

std::string SpatialMaxPooling_getKernelTemplate();

static int clnn_SpatialMaxPooling_updateOutput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  bool ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");

  THClTensor *output = (THClTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");
  THClTensor *indices = (THClTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 3, input, output, indices));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  luaL_argcheck(L, nInputCols >= kW - padW && nInputRows >= kH - padH, 2, "input image smaller than kernel size");
  luaL_argcheck(L, kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }

  input = THClTensor_newContiguous(state, input);
  //float* input_data = THClTensor_data(state, input);

  THClTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THClTensor_resizeAs(state, indices, output);
  
  //float* indices_data = THClTensor_data(state, indices);
 // float* output_data = THClTensor_data(state, output);

  int count = THClTensor_nElement(state, output);

  EasyCL *cl = input->storage->cl;
  std::string uniqueName = __FILE__ "MaxPoolForward";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("forward", 1);
    kernelBuilder.set("Dtype", "float");
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      SpatialMaxPooling_getKernelTemplate(), "MaxPoolForward");
  }

  THClKernels k(state, kernel);
  k.in((int)count);
  k.in(input);
  k.in((int)batchSize);
  k.in((int)nInputPlane);
  k.in((int)nInputRows);
  k.in((int)nInputCols);
  k.in((int)nOutputRows);
  k.in((int)nOutputCols);
  k.in((int)kH);
  k.in((int)kW);
  k.in((int)dH);
  k.in((int)dW);
  k.in((int)padH);
  k.in((int)padW);
  k.out(output);
  k.out(indices);

  dim3 blocks(GET_BLOCKS(state, count));
  dim3 threads(GET_CL_NUM_THREADS(state));
  k.run(blocks, threads);

//    maxpool <<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (
//      input_data, output_data,
//      indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
//      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
//    THError("not implemented");

  if(input->nDimension == 3)
    THClTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

  THClTensor_free(state, input);

  return 1;
}

static int clnn_SpatialMaxPooling_updateGradInput(lua_State *L)
{
  THClState *state = getCltorchState(L);
  THClTensor *input = (THClTensor *)luaT_checkudata(L, 2, "torch.ClTensor");
  THClTensor *gradOutput = (THClTensor *)luaT_checkudata(L, 3, "torch.ClTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  bool ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");

  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");
  THClTensor *indices = (THClTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 4, input, gradOutput, indices, gradInput));

  input = THClTensor_newContiguous(state, input);
  gradOutput = THClTensor_newContiguous(state, gradOutput);

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }


  gradOutput = THClTensor_newContiguous(state, gradOutput);
  THClTensor_resizeAs(state, gradInput, input);
  
  int count = THClTensor_nElement(state, input);

  EasyCL *cl = input->storage->cl;
  std::string uniqueName = __FILE__ "MaxPoolBackward";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("backward", 1);
    kernelBuilder.set("Dtype", "float");
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      SpatialMaxPooling_getKernelTemplate(), "MaxPoolBackward");
  }

  THClKernels k(state, kernel);
  k.in((int)count);
  k.in(gradOutput);
  k.in(indices);
  k.in((int)batchSize);
  k.in((int)nInputPlane);
  k.in((int)nInputRows);
  k.in((int)nInputCols);
  k.in((int)nOutputRows);
  k.in((int)nOutputCols);
  k.in((int)kH);
  k.in((int)kW);
  k.in((int)dH);
  k.in((int)dW);
  k.in((int)padH);
  k.in((int)padW);
  k.out(gradInput);

  dim3 blocks(GET_BLOCKS(state, count));
  dim3 threads(GET_CL_NUM_THREADS(state));
  k.run(blocks, threads);

//  MaxPoolBackward <<< GET_BLOCKS(count), CL_NUM_THREADS, 0, THClState_getCurrentStream(state) >>> 
//      (count,
//      THClTensor_data(state, gradOutput),
//      THClTensor_data(state, indices),
//      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
//      kH, kW, dH, dW, padH, padW,
//      THClTensor_data(state, gradInput));

  // clean
  THClTensor_free(state, input);
  THClTensor_free(state, gradOutput);

  return 1;
}

static const struct luaL_Reg clnn_SpatialMaxPooling__ [] = {
  {"SpatialMaxPooling_updateOutput", clnn_SpatialMaxPooling_updateOutput},
  {"SpatialMaxPooling_updateGradInput", clnn_SpatialMaxPooling_updateGradInput},
  {NULL, NULL}
};

void clnn_SpatialMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialMaxPooling__, "nn");
  lua_pop(L,1);
}

std::string SpatialMaxPooling_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "SpatialMaxPooling.cl" )
  // ]]]
  // generated using cog, from SpatialMaxPooling.cl:
  const char * kernelSource =  
  "// from SpatialMaxPooling.cu:\n"
  "\n"
  "// kernels borrowed from Caffe\n"
  "\n"
  "#define CL_KERNEL_LOOP(i, n)                        \\\n"
  "  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\\n"
  "      i < (n);                                       \\\n"
  "      i += get_local_size(0) * get_num_groups(0))\n"
  "\n"
  "#define Dtype {{Dtype}}\n"
  "\n"
  "\n"
  "{% if forward then %}\n"
  "kernel void MaxPoolForward(const int nthreads,\n"
  "    global const Dtype* bottom_data_data, int bottom_data_offset,\n"
  "    const int num, const int channels, const int height,\n"
  "    const int width, const int pooled_height, const int pooled_width,\n"
  "    const int kernel_h, const int kernel_w, const int stride_h,\n"
  "    const int stride_w, const int pad_h, const int pad_w,\n"
  "    global Dtype* top_data_data, int top_data_offset,\n"
  "    global Dtype* top_mask_data, int top_mask_offset\n"
  "  ) {\n"
  "\n"
  "  global const Dtype *bottom_data = bottom_data_data + bottom_data_offset;\n"
  "  global Dtype *top_data = top_data_data + top_data_offset;\n"
  "  global Dtype *top_mask = top_mask_data + top_mask_offset;\n"
  "\n"
  "  CL_KERNEL_LOOP(index, nthreads) {\n"
  "    int pw = index % pooled_width;\n"
  "    int ph = (index / pooled_width) % pooled_height;\n"
  "    int c = (index / pooled_width / pooled_height) % channels;\n"
  "    int n = index / pooled_width / pooled_height / channels;\n"
  "    int hstart = ph * stride_h - pad_h;\n"
  "    int wstart = pw * stride_w - pad_w;\n"
  "    int hend = min(hstart + kernel_h, height);\n"
  "    int wend = min(wstart + kernel_w, width);\n"
  "    hstart = max(hstart, 0);\n"
  "    wstart = max(wstart, 0);\n"
  "    Dtype maxval = -FLT_MAX;\n"
  "    int maxidx = -1;\n"
  "    bottom_data += (n * channels + c) * height * width;\n"
  "    for (int h = hstart; h < hend; ++h) {\n"
  "      for (int w = wstart; w < wend; ++w) {\n"
  "        if (bottom_data[h * width + w] > maxval) {\n"
  "          maxidx = h * width + w;\n"
  "          maxval = bottom_data[maxidx];\n"
  "        }\n"
  "      }\n"
  "    }\n"
  "    top_data[index] = maxval;\n"
  "    top_mask[index] = maxidx + 1;\n"
  "  }\n"
  "}\n"
  "{% end %}\n"
  "\n"
  "{% if backward then %}\n"
  "kernel void MaxPoolBackward(\n"
  "    const int nthreads,\n"
  "    global const Dtype* top_diff_data, int top_diff_offset,\n"
  "    global const Dtype* top_mask_data, const int top_mask_offset,\n"
  "    const int num, const int channels,\n"
  "    const int height, const int width, const int pooled_height,\n"
  "    const int pooled_width, const int kernel_h, const int kernel_w,\n"
  "    const int stride_h, const int stride_w, const int pad_h, const int pad_w,\n"
  "    global Dtype* bottom_diff_data, int bottom_diff_offset\n"
  "  ) {\n"
  "\n"
  "  global const Dtype *top_diff = top_diff_data + top_diff_offset;\n"
  "  global const Dtype *top_mask = top_mask_data + top_mask_offset;\n"
  "  global Dtype *bottom_diff = bottom_diff_data + bottom_diff_offset;\n"
  "\n"
  "  // looks like this could probably be baked, anyway, we can do that later...\n"
  "  CL_KERNEL_LOOP(index, nthreads) {\n"
  "    // find out the local index\n"
  "    // find out the local offset\n"
  "    int w = index % width;\n"
  "    int h = (index / width) % height;\n"
  "    int c = (index / width / height) % channels;\n"
  "    int n = index / width / height / channels;\n"
  "    int phstart =\n"
  "        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;\n"
  "    int phend = min((h + pad_h) / stride_h + 1, pooled_height);\n"
  "    int pwstart =\n"
  "        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;\n"
  "    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);\n"
  "    Dtype gradient = 0;\n"
  "    int offset = (n * channels + c) * pooled_height * pooled_width;\n"
  "    top_diff += offset;\n"
  "    top_mask += offset;\n"
  "    for (int ph = phstart; ph < phend; ++ph) {\n"
  "      for (int pw = pwstart; pw < pwend; ++pw) {\n"
  "	if (top_mask[ph * pooled_width + pw] - 1 == h * width + w) {\n"
  "	  gradient += top_diff[ph * pooled_width + pw];\n"
  "	}\n"
  "      }\n"
  "    }\n"
  "    bottom_diff[index] = gradient;\n"
  "  }\n"
  "}\n"
  "{% end %}\n"
  "";
  // [[[end]]]
  return kernelSource;
}

