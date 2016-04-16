// Use 1024 threads per block, which requires cuda sm_2x or above
//const int CL_NUM_THREADS = 1024;

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"
#include "DeviceInfo.h"
#include "EasyCL.h"
#include "im2col.h"

static std::string getKernelTemplate();

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

// im is the incoming image; col is the outoing unrolled matrix
// it's something like:
// columns is [inPlanes * kH * kW][outH * outW]
// weight is [outplanes][inplanes * kH * kW]
// output is [outPlanes][outH][outW]
// (but might be transposed a bit)
void im2col(THClState *state, THClTensor* im, const int channels,
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH, 
    const int padW, const int padH,
    THClTensor* col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
//  int width_col = (width + 2 * padW - ksize_w) / dW + 1;
//  int height_col = (height + 2 * padH - ksize_h) / dH + 1;
  // seems like height_col and width_col are just output width/height?
  long outW  = (inputWidth + 2*padW - kW) / dW + 1;
  long outH = (inputHeight + 2*padH - kH) / dH + 1;

  int num_kernels = channels * height_col * width_col;

  std::string uniqueName = "SpatialConvolutionMM::im2col";
  EasyCL *cl = im->storage->cl;
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialConvolutionMM.cl",
      getKernelTemplate(), "im2col_kernel");
  }

  THClKernels k(state, kernel);
  k.in(num_kernels);
  k.in(im);
  k.in(inW);
  k.in(inH);
  k.in(kW);
  k.in(kH);
  k.in(dW);
  k.in(dH);
  k.in(padW);
  k.in(padH);
  k.in(outW);
  k.in(outH);
  k.out(col);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}

void col2im(THClState *state, THClTensor* col,
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    THClTensor* im) {
  int outW = (inW + 2 * padW - kW) / dW + 1;
  int outH = (inH + 2 * padH - kH) / dH + 1;
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
      getKernelTemplate(), "col2im_kernel");
  }

  THClKernels k(state, kernel);
  k.in(num_kernels);
  k.in(col);
  k.in(inW);
  k.in(inH);

  k.in(kW);
  k.in(kH);
  k.in(dW);
  k.in(dH);
  k.in(padW);
  k.in(padH);

  k.in(outW);
  k.in(outH);
  k.out(im);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}

#undef CL_KERNEL_LOOP

std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "lib/THCLNN/im2col.cl" )
  // ]]]
  // generated using cog, from lib/THCLNN/im2col.cl:
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
  "    const int inW, const int inH,\n"
  "     const int kW, const int kH,\n"
  "    const int dW, const int dH,\n"
  "    const int padW, const int padH,\n"
  "    const int outW, const int outH,\n"
  "    global float* col_data, int col_offset) {\n"
  "  global const float *data_im = im_data + im_offset;\n"
  "  global float *data_col = col_data + col_offset;\n"
  "\n"
  "  CL_KERNEL_LOOP(index, n) {\n"
  "    int w_out = index % outW;\n"
  "    index /= outW;\n"
  "    int h_out = index % outH;\n"
  "    int channel_in = index / outH;\n"
  "    int channel_out = channel_in * kW * kH;\n"
  "    int w_in = w_out * dW - padW;\n"
  "    int h_in = h_out * dH - padH;\n"
  "    data_col += (channel_out * outH + h_out) * outW + w_out;\n"
  "    data_im += (channel_in * inH + h_in) * inW + w_in;\n"
  "    for (int i = 0; i < kH; ++i) {\n"
  "      for (int j = 0; j < kW; ++j) {\n"
  "        int h = h_in + i;\n"
  "        int w = w_in + j;\n"
  "        *data_col = (h >= 0 && w >= 0 && h < inH && w < inW) ?\n"
  "          data_im[i * inW + j] : 0;\n"
  "        data_col += outH * outW;\n"
  "      }\n"
  "    }\n"
  "  }\n"
  "}\n"
  "\n"
  "// Kernel for fast unfold+copy\n"
  "// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)\n"
  "// adapted for use with groups, as described in\n"
  "// \"OpenCL caffe: Accelerating and enabling a cross platform machine\" by Junli Gu et al\n"
  "// imageIndex will control where in columns, the image is copied, and columns will have in fact\n"
  "// numimages *\n"
  "//kernel void im2col_grouped_kernel(const int n, const global float* im_data, int im_offset,\n"
  "//    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,\n"
  "//    const int pad_w, const int dH, const int dW, const int height_col, const int width_col,\n"
  "//    global float* col_data, int col_offset, int numImages, int imageIndex) {\n"
  "//  global const float *data_im = im_data + im_offset;\n"
  "//  global float *data_col = col_data + col_offset;\n"
  "\n"
  "//  CL_KERNEL_LOOP(index, n) {\n"
  "//    int w_out = index % width_col;\n"
  "//    index /= width_col;\n"
  "//    int h_out = index % height_col;\n"
  "//    int channel_in = index / height_col;\n"
  "//    int channel_out = channel_in * ksize_h * ksize_w;\n"
  "//    int h_in = h_out * dH - pad_h;\n"
  "//    int w_in = w_out * dW - pad_w;\n"
  "//    data_col += (channel_out * height_col + h_out) * width_col + w_out;\n"
  "//    data_im += (channel_in * height + h_in) * width + w_in;\n"
  "//    for (int i = 0; i < ksize_h; ++i) {\n"
  "//      for (int j = 0; j < ksize_w; ++j) {\n"
  "//        int h = h_in + i;\n"
  "//        int w = w_in + j;\n"
  "//        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?\n"
  "//          data_im[i * width + j] : 0;\n"
  "//        data_col += height_col * width_col;\n"
  "//      }\n"
  "//    }\n"
  "//  }\n"
  "//}\n"
  "\n"
  "kernel void col2im_kernel(const int n, global const float* col_data, int col_offset,\n"
  "    const int inW, const int inH,\n"
  "    const int kW, const int kH,\n"
  "    const int dW, const int dH,\n"
  "    const int padW, const int padH,\n"
  "    const int outW, const int outH,\n"
  "    global float* im_data, int im_offset) {\n"
  "  global float *data_im = im_data + im_offset;\n"
  "  global const float *data_col = col_data + col_offset;\n"
  "\n"
  "  CL_KERNEL_LOOP(index, n) {\n"
  "    float val = 0;\n"
  "    int w = index % inW + padW;\n"
  "    int h = (index / inW) % inH + padH;\n"
  "    int c = index / (inW * inH);\n"
  "    // compute the start and end of the output\n"
  "    int w_col_start = (w < kW) ? 0 : (w - kW) / dW + 1;\n"
  "    int w_col_end = min(w / dW + 1, outW);\n"
  "    int h_col_start = (h < kH) ? 0 : (h - kH) / dH + 1;\n"
  "    int h_col_end = min(h / dH + 1, outH);\n"
  "    /*\n"
  "       for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n"
  "       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n"
  "    // the col location: [c * width * height + h_out, w_out]\n"
  "    int c_col = c * patch_h * patch_w + (h - h_col * dH) * ksize + (w - w_col * dW);\n"
  "    val += data_col[(c_col * height_col + h_col) * width_col + w_col];\n"
  "    }\n"
  "    }\n"
  "     */\n"
  "    // equivalent implementation\n"
  "    int offset = (c * kH * kW + h * kW + w) * outH * outW;\n"
  "    int coeff_h_col = (1 - dH * kW * outH) * outW;\n"
  "    int coeff_w_col = (1 - dW * outH * outW);\n"
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

