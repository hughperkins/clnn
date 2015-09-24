// from SpatialAveragePooling.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "common.h"

#include "EasyCL.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"

#include <iostream>
#include <string>
using namespace std;

static std::string getKernelTemplate();

static int clnn_SpatialAveragePooling_updateOutput(lua_State *L)
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
  bool count_include_pad = luaT_getfieldcheckboolean(L, 1, "count_include_pad");

  THClTensor *output = (THClTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 2, input, output));
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
  // input_data = THClTensor_data(state, input);

  THClTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  
  // float* output_data = THClTensor_data(state, output);

  int count = THClTensor_nElement(state, output);

  EasyCL *cl = input->storage->cl;
  const char *uniqueName = 0;
  if(count_include_pad) {
    uniqueName = __FILE__ "forward_includepad";
  } else {
    uniqueName = __FILE__ "forward_not_includepad";
  }
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("forward", 1);
    kernelBuilder.set("Dtype", "float");
    kernelBuilder.set("COUNT_INCLUDE_PAD", count_include_pad);
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      getKernelTemplate(), "AvePoolForward");
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

  dim3 blocks(GET_BLOCKS(state, count));
  dim3 threads(GET_CL_NUM_THREADS(state));
  k.run(blocks, threads);

//    AvePoolForward<float, true>
//      <<<GET_BLOCKS(count), CL_NUM_THREADS, 0, THClState_getCurrentStream(state) >>>(
//        count, input_data,
//        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
//        kH, kW, dH, dW, padH, padW, output_data);

  if(input->nDimension == 3)
    THClTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

  THClTensor_free(state, input);

  return 1;
}

static int clnn_SpatialAveragePooling_updateGradInput(lua_State *L)
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
  bool count_include_pad = luaT_getfieldcheckboolean(L, 1, "count_include_pad");

  THClTensor *gradInput = (THClTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.ClTensor");

  THAssert(THClTensor_checkGPU(state, 3, input, gradOutput, gradInput));

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
  const char *uniqueName = 0;
  if(count_include_pad) {
    uniqueName = __FILE__ "backward_includepad";
  } else {
    uniqueName = __FILE__ "backward_not_includepad";
  }
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("backward", 1);
    kernelBuilder.set("Dtype", "float");
    kernelBuilder.set("COUNT_INCLUDE_PAD", count_include_pad);
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      getKernelTemplate(), "AvePoolBackward");
  }

  THClKernels k(state, kernel);
  k.in((int)count);
  k.in(gradOutput);
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

//    AvePoolBackward<float, true>
//      <<< GET_BLOCKS(count), CL_NUM_THREADS, 0, THClState_getCurrentStream(state) >>> 
//        (count,
//        THClTensor_data(state, gradOutput),
//        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
//        kH, kW, dH, dW, padH, padW,
//        THClTensor_data(state, gradInput));

  THClTensor_free(state, gradOutput);

  // clean
  THClTensor_free(state, input);
  THClTensor_free(state, gradOutput);

  return 1;
}


static const struct luaL_Reg clnn_SpatialAveragePooling__ [] = {
  {"SpatialAveragePooling_updateOutput", clnn_SpatialAveragePooling_updateOutput},
  {"SpatialAveragePooling_updateGradInput", clnn_SpatialAveragePooling_updateGradInput},
  {NULL, NULL}
};

void clnn_SpatialAveragePooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialAveragePooling__, "nn");
  lua_pop(L,1);
}

//#undef CL_MAX_THREADS

static std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "SpatialAveragePooling.cl" )
  // ]]]
  // generated using cog, from SpatialAveragePooling.cl:
  const char * kernelSource =  
  "// from SpatialAveragePooling.cu:\n" 
  "\n" 
  "#define Dtype {{Dtype}}\n" 
  "#define COUNT_INCLUDE_PAD {{COUNT_INCLUDE_PAD}}\n" 
  "\n" 
  "#define CL_KERNEL_LOOP(i, n)                        \\\n" 
  "  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\\n" 
  "      i < (n);                                       \\\n" 
  "      i += get_local_size(0) * get_num_groups(0))\n" 
  "\n" 
  "{% if forward then %}\n" 
  "kernel void AvePoolForward(const int nthreads,\n" 
  "    global const Dtype* const bottom_data_data, int bottom_data_offset,\n" 
  "    const int num, const int channels,\n" 
  "    const int height, const int width, const int pooled_height,\n" 
  "    const int pooled_width, const int kernel_h, const int kernel_w,\n" 
  "    const int stride_h, const int stride_w, const int pad_h, const int pad_w,\n" 
  "    global Dtype* const top_data_data, int top_data_offset\n" 
  "    ) {\n" 
  "  global const Dtype* const bottom_data = bottom_data_data + bottom_data_offset;\n" 
  "  global Dtype* const top_data = top_data_data + top_data_offset;\n" 
  "\n" 
  "  // bake in later, once working and if this layer is shown to contribute highly to\n" 
  "  // slowness\n" 
  "  CL_KERNEL_LOOP(index, nthreads) {\n" 
  "    const int pw = index % pooled_width;\n" 
  "    const int ph = (index / pooled_width) % pooled_height;\n" 
  "    const int c = (index / pooled_width / pooled_height) % channels;\n" 
  "    const int n = index / pooled_width / pooled_height / channels;\n" 
  "    int hstart = ph * stride_h - pad_h;\n" 
  "    int wstart = pw * stride_w - pad_w;\n" 
  "    int hend = min(hstart + kernel_h, height + pad_h);\n" 
  "    int wend = min(wstart + kernel_w, width + pad_w);\n" 
  "    const int pool_size = (hend - hstart) * (wend - wstart);\n" 
  "    hstart = max(hstart, 0);\n" 
  "    wstart = max(wstart, 0);\n" 
  "    hend = min(hend, height);\n" 
  "    wend = min(wend, width);\n" 
  "    Dtype aveval = 0;\n" 
  "    global const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;\n" 
  "    for (int h = hstart; h < hend; ++h) {\n" 
  "      for (int w = wstart; w < wend; ++w) {\n" 
  "        aveval += bottom_slice[h * width + w];\n" 
  "      }\n" 
  "    }\n" 
  "    if(COUNT_INCLUDE_PAD)\n" 
  "      top_data[index] = aveval / ((hend - hstart) * (wend - wstart));\n" 
  "    else\n" 
  "      top_data[index] = aveval / pool_size;\n" 
  "  }\n" 
  "}\n" 
  "{% end %}\n" 
  "\n" 
  "{% if backward then %}\n" 
  "kernel void AvePoolBackward(\n" 
  "    const int nthreads,\n" 
  "    global const Dtype* const top_diff_data, int top_diff_offset,\n" 
  "    const int num, const int channels, const int height,\n" 
  "    const int width, const int pooled_height, const int pooled_width,\n" 
  "    const int kernel_h, const int kernel_w, const int stride_h,\n" 
  "    const int stride_w, const int pad_h, const int pad_w,\n" 
  "    global Dtype* const bottom_diff_data, int bottom_diff_offset\n" 
  "    ) {\n" 
  "  global const Dtype * const top_diff = top_diff_data + top_diff_offset;\n" 
  "  global Dtype *const bottom_diff = bottom_diff_data + bottom_diff_offset;\n" 
  "  CL_KERNEL_LOOP(index, nthreads) {\n" 
  "    // find out the local index\n" 
  "    // find out the local offset\n" 
  "    const int w = index % width + pad_w;\n" 
  "    const int h = (index / width) % height + pad_h;\n" 
  "    const int c = (index / width / height) % channels;\n" 
  "    const int n = index / width / height / channels;\n" 
  "    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;\n" 
  "    const int phend = min(h / stride_h + 1, pooled_height);\n" 
  "    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;\n" 
  "    const int pwend = min(w / stride_w + 1, pooled_width);\n" 
  "    Dtype gradient = 0;\n" 
  "    global const Dtype* const top_diff_slice =\n" 
  "        top_diff + (n * channels + c) * pooled_height * pooled_width;\n" 
  "    for (int ph = phstart; ph < phend; ++ph) {\n" 
  "      for (int pw = pwstart; pw < pwend; ++pw) {\n" 
  "        // figure out the pooling size\n" 
  "        int hstart = ph * stride_h - pad_h;\n" 
  "        int wstart = pw * stride_w - pad_w;\n" 
  "        int hend = min(hstart + kernel_h, height + pad_h);\n" 
  "        int wend = min(wstart + kernel_w, width + pad_w);\n" 
  "        int pool_size = (hend - hstart) * (wend - wstart);\n" 
  "        if(COUNT_INCLUDE_PAD)\n" 
  "          gradient += top_diff_slice[ph * pooled_width + pw] / ((hend - hstart) * (wend - wstart));\n" 
  "        else\n" 
  "          gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;\n" 
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
