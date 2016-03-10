// from SpatialUpSamplingNearest.cu:


#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THClBlas.h"
#include "EasyCL.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"
#include "common.h"
#include "THCLNN.h"

#include <iostream>
#include <string>
using namespace std;


// #include <thrust/transform.h>
// #include <thrust/reduce.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/functional.h>


/*
 * Description:
 */

std::string SpatialUpSamplingNearest_getKernelTemplate();

void THNN_ClSpatialUpSamplingNearest_updateOutput(THClState *state, THClTensor *input, THClTensor *output, int scale_factor)
{

  THAssert(THClTensor_checkGPU(state, 2, input, output));

  input = THClTensor_newContiguous(state, input);
  output = THClTensor_newContiguous(state, output);
  // This is for allocating output Tensor
  long no_elements = 1;
  for(int i = 0; i < input->nDimension; i++){
    no_elements *= input->size[i];
  }
  no_elements *= scale_factor * scale_factor;

  int d1;
  int d2;
  int d3;
  if (input->nDimension == 3) {

    d1 = output->size[0];
    d2 = output->size[1];
    d3 = output->size[2];
  } else {

    d1 = output->size[1];
    d2 = output->size[2];
    d3 = output->size[3];
  }

  //float *input_data = THClTensor_data(state, input);
  //float *output_data = THClTensor_data(state, output);

  int count = THClTensor_nElement(state, output);
  EasyCL *cl = input->storage->cl;
  std::string uniqueName = "SpatialUpSamplingNearest::upscale";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialUpSamplingNearest.cl",
      SpatialUpSamplingNearest_getKernelTemplate(), "upscale");
  }



  THClKernels k(state, kernel);
  k.in(input);
  k.out(output);
  k.in((int)no_elements);
  k.in((int)scale_factor);
  k.in((int)d1);
  k.in((int)d2);
  k.in((int)d3);

  k.run(GET_BLOCKS(state, count), GET_CL_NUM_THREADS(state));
  // kernel:
  //upscale<blocks, threads, 0, THClState_getCurrentStream(state)> (input_data, output_data, no_elements, scale_factor, d1, d2, d3);

  // check for errors

  // final cut:
  THClTensor_free(state, input);

}


void THNN_ClSpatialUpSamplingNearest_updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradInput, int scale_factor)
{
  THAssert(THClTensor_checkGPU(state, 2, gradOutput, gradInput));

  THClTensor_zero(state, gradInput);

  //float *gradInput_data = THClTensor_data(state, gradInput);
  //float *gradOutput_data = THClTensor_data(state, gradOutput);

  gradInput = THClTensor_newContiguous(state, gradInput);
  gradOutput = THClTensor_newContiguous(state, gradOutput);

  long no_elements = 1;
  for(int i = 0; i < gradInput->nDimension; i++){
    no_elements *= gradInput->size[i];
  }

  int d1;
  int d2;
  int d3;

  if (gradInput->nDimension == 3) {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
    d3 = gradInput->size[2];
  } else {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
    d3 = gradInput->size[3];
  }

  int count = THClTensor_nElement(state, gradInput);

  EasyCL *cl = gradInput->storage->cl;
  std::string uniqueName = "SpatialUpSamplingNearest::downscale";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialUpSamplingNearest.cl",
      SpatialUpSamplingNearest_getKernelTemplate(), "downscale");
  }
  //std::cout << gradInput << std::endl;
  THClKernels k(state, kernel);
  k.out(gradInput);
  k.in(gradOutput);
  k.in((int)no_elements);
  k.in((int)scale_factor);
  k.in((int)d1);
  k.in((int)d2);
  k.in((int)d3);

  k.run(GET_BLOCKS(state, count), GET_CL_NUM_THREADS(state));
  //std::cout << gradOutput << std::endl;
  // kernel:
  // TODO: kernel
  /* downscale<<<blocks, threads, 0, THClState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_elements,
    scale_factor, d1, d2, d3);
  */
  // check for errors
  THClTensor_free(state, gradOutput);

}
/*
const struct luaL_Reg clnn_SpatialUpSamplingNearest__ [] = {
  {"SpatialUpSamplingNearest_updateOutput", clnn_SpatialUpSamplingNearest_updateOutput},
  {"SpatialUpSamplingNearest_updateGradInput", clnn_SpatialUpSamplingNearest_updateGradInput},
  {NULL, NULL}
};

void clnn_SpatialUpSamplingNearest_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaT_registeratname(L, clnn_SpatialUpSamplingNearest__, "nn");
  lua_pop(L,1);
}*/

std::string SpatialUpSamplingNearest_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "lib/THCLNN/SpatialUpSamplingNearest.cl" )
  // ]]]
  // generated using cog, from lib/THCLNN/SpatialUpSamplingNearest.cl:
  const char * kernelSource =  
  "// from SpatialUpSamplingNearest.cu:\n"
  "\n"
  "/*__device__*/ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor)\n"
  "{\n"
  "  int x, y, z, w;\n"
  "  w = ii % d3;\n"
  "  ii = ii/d3;\n"
  "  z = ii % d2;\n"
  "  ii = ii/d2;\n"
  "  y = ii % d1;\n"
  "  ii = ii/d1;\n"
  "  x = ii;\n"
  "  w = w/scale_factor;\n"
  "  z = z/scale_factor;\n"
  "  d2 /= scale_factor;\n"
  "  d3 /= scale_factor;\n"
  "  return (((x*d1+y)*d2)+z)*d3+w;\n"
  "\n"
  "}\n"
  "/*__device__*/ int translate_idx_inv(int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y)\n"
  "{\n"
  "  int x, y, z, w;\n"
  "  w = ii % d3;\n"
  "  ii = ii/d3;\n"
  "  z = ii % d2;\n"
  "  ii = ii/d2;\n"
  "  y = ii % d1;\n"
  "  ii = ii/d1;\n"
  "  x = ii;\n"
  "  w = w*scale_factor+off_x;\n"
  "  z = z*scale_factor+off_y;\n"
  "  d2 *= scale_factor;\n"
  "  d3 *= scale_factor;\n"
  "  return (((x*d1+y)*d2)+z)*d3+w;\n"
  "\n"
  "}\n"
  "\n"
  "kernel void upscale(global float *input_data, int input_offset, global float *output_data, int output_offset, int no_elements,\n"
  "                        int scale_factor, int d1, int d2, int d3)\n"
  "{\n"
  "  global float *input = input_data + input_offset;\n"
  "  global float *output = output_data + output_offset;\n"
  "  // output offset:\n"
  "  long ii = get_local_id(0) + get_local_size(0) * get_group_id(0);\n"
  "  ii += get_local_id(1) + get_local_size(1) * (get_local_size(0) * get_num_groups(0)) * get_group_id(1);\n"
  "  if (ii >= no_elements) return;\n"
  "  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);\n"
  "  output[ii]=input[ipidx];\n"
  "}\n"
  "\n"
  "/*\n"
  " * Description:\n"
  " */\n"
  "kernel void downscale(global float *gradInput_data_data, int gradInput_data_offset, global float *gradOutput_data_data, int gradOutput_data_offset, long no_elements,\n"
  "                              int scale_factor, int d1, int d2, int d3)\n"
  "{\n"
  "  global float *gradInput_data = gradInput_data_data + gradInput_data_offset;\n"
  "  global float *gradOutput_data = gradOutput_data_data + gradOutput_data_offset;\n"
  "  // output offset:\n"
  "  long ii = get_local_id(0) + get_local_size(0) * get_group_id(0);\n"
  "  ii += get_local_id(1) + get_local_size(1) * (get_local_size(0) * get_num_groups(0)) * get_group_id(1);\n"
  "  if (ii >= no_elements) return;\n"
  "  for (int i=0; i < scale_factor; i++){\n"
  "    for(int j=0; j < scale_factor; j++){\n"
  "      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);\n"
  "      gradInput_data[ii] += gradOutput_data[ipidx];\n"
  "    }\n"
  "  }\n"
  "}\n"
  "";
  // [[[end]]]
  return kernelSource;
}
