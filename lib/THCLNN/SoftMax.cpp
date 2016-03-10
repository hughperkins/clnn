// from SoftMax.cu:

#include "utils.h"
#include "luaT.h"
#include "THClTensor.h"
#include "THClTensorMath.h"
#include "THCLNN.h"

#include "EasyCL.h"
#include "THClKernels.h"
#include "templates/TemplatedKernel.h"

#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAX_THREADS 128  // compatible with AMD too, so I guess we just use it?

static std::string getKernelTemplate();

void THNN_ClSoftMax_updateOutput(THClState *state, THClTensor *input, THClTensor *output)
{
  THAssert(THClTensor_checkGPU(state, 2, input, output));

  input = THClTensor_newContiguous(state, input);
  THClTensor_resizeAs(state, output, input);
  long batchSize = 0;
  long dim = 0;
  long stride = 0;

  if (input->nDimension == 1)
  {
    batchSize = 1;
    dim = input->size[0];
    stride = 1;
  }
  else if (input->nDimension == 2)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride = 1;
  }
  else if (input->nDimension == 3)
  {
    batchSize = 1;
    dim = input->size[0];
    stride = input->size[1] * input->size[2];
  }
  else if (input->nDimension == 4)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride = input->size[2] * input->size[3];
  }
  else
    THError("1D, 2D, 3D or 4D tensor expected");

  EasyCL *cl = input->storage->cl;
  const char *uniqueName = __FILE__ "forward";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("forward", 1);
    kernelBuilder.set("SOFTMAX_THREADS", SOFTMAX_THREADS);
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      getKernelTemplate(), "updateOutput");
  }

  THClKernels k(state, kernel);
  k.out(output);
  k.in(input);
  k.in((int)batchSize);
  k.in((int)dim);
  k.in((int)stride);

  dim3 blocks(batchSize, stride);
  dim3 threads(SOFTMAX_THREADS);
  k.run(blocks, threads);

//  clnn_SoftMax_updateOutput_kernel<<<blocks,threads,
//    0, THClState_getCurrentStream(state)>>>(THClTensor_data(state, output),
//                                           THClTensor_data(state, input),
//                                           batchSize, dim, stride);

  THClTensor_free(state, input);
}

void THNN_ClSoftMax_updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradInput, THClTensor *output)
{
  THAssert(THClTensor_checkGPU(state, 3, output, gradOutput, gradInput));

  output = THClTensor_newContiguous(state, output);
  gradOutput = THClTensor_newContiguous(state, gradOutput);

  THClTensor_resizeAs(state, gradInput, output);
  long batchSize = 0;
  long dim = 0;
  long stride = 0;

  if (gradInput->nDimension == 1)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride = 1;
  }
  else if (gradInput->nDimension == 2)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride = 1;
  }
  else if (gradInput->nDimension == 3)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride = gradInput->size[1] * gradInput->size[2];
  }
  else if (gradInput->nDimension == 4)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride = gradInput->size[2] * gradInput->size[3];
  }
  else
    THError("1D, 2D, 3D or 4D tensor expected");

  EasyCL *cl = gradOutput->storage->cl;
  const char *uniqueName = __FILE__ "backward";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("backward", 1);
    kernelBuilder.set("SOFTMAX_THREADS", SOFTMAX_THREADS);
    kernel = kernelBuilder.buildKernel(uniqueName, __FILE__,
      getKernelTemplate(), "updateGradInput");
  }

  THClKernels k(state, kernel);
  k.out(gradInput);
  k.in(output);
  k.in(gradOutput);
  k.in((int)batchSize);
  k.in((int)dim);
  k.in((int)stride);

  dim3 blocks(batchSize, stride);
  dim3 threads(SOFTMAX_THREADS);

  k.run(blocks, threads);

//  clnn_SoftMax_updateGradInput_kernel<<<blocks,threads,
//    0, THClState_getCurrentStream(state)>>>(THClTensor_data(state, gradInput),
//                                           THClTensor_data(state, output),
//                                           THClTensor_data(state, gradOutput),
//                                           batchSize, dim, stride);

  THClTensor_free(state, gradOutput);
  THClTensor_free(state, output);
}

static std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "lib/THCLNN/SoftMax.cl" )
  // ]]]
  // generated using cog, from lib/THCLNN/SoftMax.cl:
  const char * kernelSource =  
  "// from SoftMax.cu:\n"
  "\n"
  "#define SOFTMAX_THREADS {{SOFTMAX_THREADS}}\n"
  "\n"
  "{% if forward then %}\n"
  "kernel void updateOutput(\n"
  "  global float *output_data, int output_offset,\n"
  "  global float *input_data, int input_offset,\n"
  "  int nframe, int dim, int stride)\n"
  "{\n"
  "  global float *output = output_data + output_offset;\n"
  "  global float *input = input_data + input_offset;\n"
  "\n"
  "  local float buffer[SOFTMAX_THREADS+1];\n"
  "\n"
  "  global float *input_k = input + get_group_id(0)*dim*stride + get_group_id(1);\n"
  "  global float *output_k = output + get_group_id(0)*dim*stride + get_group_id(1);\n"
  "\n"
  "  int i_start = get_local_id(0);\n"
  "  int i_end = dim;\n"
  "  int i_step = get_local_size(0);\n"
  "\n"
  "  // max?\n"
  "  buffer[get_local_id(0)] = -FLT_MAX;\n"
  "  for (int i=i_start; i<i_end; i+=i_step)\n"
  "  {\n"
  "    float z = input_k[i*stride];\n"
  "    if(buffer[get_local_id(0)] < z)\n"
  "      buffer[get_local_id(0)] = z;\n"
  "  }\n"
  "\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "\n"
  "//  // reduce\n"
  "  if (get_local_id(0) == 0)\n"
  "  {\n"
  "    float max_k = -FLT_MAX;\n"
  "    for (int i=0; i<(int)get_local_size(0); i++)\n"
  "    {\n"
  "      if(max_k < buffer[i])\n"
  "        max_k = buffer[i];\n"
  "    }\n"
  "    buffer[SOFTMAX_THREADS] = max_k;\n"
  "  }\n"
  "\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "\n"
  "//  // sum?\n"
  "  float max_k = buffer[SOFTMAX_THREADS];\n"
  "  buffer[get_local_id(0)] = 0;\n"
  "  for (int i=i_start; i<i_end; i+=i_step) {\n"
  "    float z = native_exp(input_k[i*stride]-max_k);\n"
  "    buffer[get_local_id(0)] += z;\n"
  "    output_k[i*stride] = z;\n"
  "  }\n"
  "\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "\n"
  "//  // reduce\n"
  "  if (get_local_id(0) == 0)\n"
  "  {\n"
  "    float sum_k = 0;\n"
  "    for (int i=0; i<(int)get_local_size(0); i++)\n"
  "      sum_k += buffer[i];\n"
  "    buffer[SOFTMAX_THREADS] = sum_k;\n"
  "  }\n"
  "\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "\n"
  "  // softmax\n"
  "  float sum_k = buffer[SOFTMAX_THREADS];\n"
  "  for (int i=i_start; i<i_end; i+=i_step)\n"
  "    output_k[i*stride] = output_k[i*stride] / sum_k;\n"
  "}\n"
  "{% end %}\n"
  "\n"
  "{% if backward then %}\n"
  "kernel void updateGradInput(\n"
  "  global float *gradInput_data, int gradInput_offset,\n"
  "  global float *output_data, int output_offset,\n"
  "  global float *gradOutput_data, int gradOutput_offset,\n"
  "  int nframe, int dim, int stride)\n"
  "{\n"
  "  global float *gradInput = gradInput_data + gradInput_offset;\n"
  "  global float *output = output_data + output_offset;\n"
  "  global float *gradOutput = gradOutput_data + gradOutput_offset;\n"
  "\n"
  "  local float buffer[SOFTMAX_THREADS];\n"
  "  global float *gradInput_k = gradInput + get_group_id(0)*dim*stride + get_group_id(1);\n"
  "  global float *output_k = output + get_group_id(0)*dim*stride + get_group_id(1);\n"
  "  global float *gradOutput_k = gradOutput + get_group_id(0)*dim*stride + get_group_id(1);\n"
  "\n"
  "  int i_start = get_local_id(0);\n"
  "  int i_end = dim;\n"
  "  int i_step = get_local_size(0);\n"
  "\n"
  "  // sum?\n"
  "  buffer[get_local_id(0)] = 0;\n"
  "  for (int i=i_start; i<i_end; i+=i_step)\n"
  "    buffer[get_local_id(0)] += gradOutput_k[i*stride] * output_k[i*stride];\n"
  "\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "\n"
  "  // reduce\n"
  "  if (get_local_id(0) == 0)\n"
  "  {\n"
  "    float sum_k = 0;\n"
  "    for (int i=0; i<(int)get_local_size(0); i++)\n"
  "      sum_k += buffer[i];\n"
  "    buffer[0] = sum_k;\n"
  "  }\n"
  "\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "\n"
  "  float sum_k = buffer[0];\n"
  "  for (int i=i_start; i<i_end; i+=i_step)\n"
  "    gradInput_k[i*stride] = output_k[i*stride] * (gradOutput_k[i*stride] - sum_k);\n"
  "}\n"
  "{% end %}\n"
  "\n"
  "";
  // [[[end]]]
  return kernelSource;
}

