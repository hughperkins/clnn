// from SoftMax.cu:

#define SOFTMAX_THREADS {{SOFTMAX_THREADS}}

{% if forward then %}
kernel void updateOutput(
  global float *output_data, int output_offset,
  global float *input_data, int input_offset,
  int nframe, int dim, int stride)
{
  global float *output = output_data + output_offset;
  global float *input = input_data + input_offset;

  local float buffer[SOFTMAX_THREADS+1];

  global float *input_k = input + get_group_id(0)*dim*stride + get_group_id(1);
  global float *output_k = output + get_group_id(0)*dim*stride + get_group_id(1);

  int i_start = get_local_id(0);
  int i_end = dim;
  int i_step = get_local_size(0);

  // max?
  buffer[get_local_id(0)] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i*stride];
    if(buffer[get_local_id(0)] < z)
      buffer[get_local_id(0)] = z;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

//  // reduce
  if (get_local_id(0) == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<(int)get_local_size(0); i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

//  // sum?
  float max_k = buffer[SOFTMAX_THREADS];
  buffer[get_local_id(0)] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    float z = native_exp(input_k[i*stride]-max_k);
    buffer[get_local_id(0)] += z;
    output_k[i*stride] = z;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

//  // reduce
  if (get_local_id(0) == 0)
  {
    float sum_k = 0;
    for (int i=0; i<(int)get_local_size(0); i++)
      sum_k += buffer[i];
    buffer[SOFTMAX_THREADS] = sum_k;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // softmax
  float sum_k = buffer[SOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i*stride] = output_k[i*stride] / sum_k;
}
{% end %}

{% if backward then %}
kernel void updateGradInput(
  global float *gradInput_data, int gradInput_offset,
  global float *output_data, int output_offset,
  global float *gradOutput_data, int gradOutput_offset,
  int nframe, int dim, int stride)
{
  global float *gradInput = gradInput_data + gradInput_offset;
  global float *output = output_data + output_offset;
  global float *gradOutput = gradOutput_data + gradOutput_offset;

  local float buffer[SOFTMAX_THREADS];
  global float *gradInput_k = gradInput + get_group_id(0)*dim*stride + get_group_id(1);
  global float *output_k = output + get_group_id(0)*dim*stride + get_group_id(1);
  global float *gradOutput_k = gradOutput + get_group_id(0)*dim*stride + get_group_id(1);

  int i_start = get_local_id(0);
  int i_end = dim;
  int i_step = get_local_size(0);

  // sum?
  buffer[get_local_id(0)] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[get_local_id(0)] += gradOutput_k[i*stride] * output_k[i*stride];

  barrier(CLK_LOCAL_MEM_FENCE);

  // reduce
  if (get_local_id(0) == 0)
  {
    float sum_k = 0;
    for (int i=0; i<(int)get_local_size(0); i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float sum_k = buffer[0];
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i*stride] = output_k[i*stride] * (gradOutput_k[i*stride] - sum_k);
}
{% end %}

