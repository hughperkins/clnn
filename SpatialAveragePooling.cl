// from SpatialAveragePooling.cu:

/*
 * Description:
 *    this function avg-pools an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output
 */
kernel void subsample(
  const global float *input_data, int input_offset,
  global float *output_data, int output_offset,
  int input_n, int input_h, int input_w,
  int kH, int kW, int dH, int dW)
{
  global const float *input = input_data + input_offset;
  global float *output = output_data + output_offset;

  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = get_group_id(0);
  int i = o;

  int xx_start = get_local_id(0);
  int xx_end = output_w;
  int xx_step = get_local_size(0);

  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);
  int yy_end = output_h;
  int yy_step = get_local_size(1)*get_num_groups(1);

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      // Compute the mean of the input image...
      const global float *ptr_input = input + yy*dH*input_w + xx*dW;
      global float *ptr_output = output + yy*output_w + xx;
      float sum = 0;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          sum += ptr_input[kx];
        ptr_input += input_w; // next input line
      }
      // Update output
      *ptr_output = sum/(float)(kW*kH);
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from gradOutput
 */
kernel void subgradinput(
  global float *gradInput_data, int gradInput_offset,
  const global float *gradOutput_data, int gradOutput_offset,
  int input_n, int input_h, int input_w,
  int kH, int kW, int dH, int dW)
{
  global float *gradInput = gradInput_data + gradInput_offset;
  const global float *gradOutput = gradOutput_data + gradOutput_offset;

  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = get_group_id(0);
  int i = o;

  int xx_start = get_local_id(0);
  int xx_end = output_w;
  int xx_step = get_local_size(0);

  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);
  int yy_end = output_h;
  int yy_step = get_local_size(1)*get_num_groups(1);

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      global float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      const global float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          ptr_gradInput[kx] += z / (float)(kW*kH);
        ptr_gradInput += input_w;
      }
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from gradOutput
 *    but with an atomic accumulation. It is needed to be done so
 *    for cases of kH != dH and kW != dW
 */
//kernel void subgradinputAtomic(float *gradInput, float *gradOutput,
//                                   int input_n, int input_h, int input_w,
//                                   int kH, int kW, int dH, int dW)
//{
//  // iterators
//  int xx, yy;

//  // output size
//  int output_w = (input_w - kW) / dW + 1;
//  int output_h = (input_h - kH) / dH + 1;

//  // compute offsets based on thread/block ID
//  int o = get_group_id(0);
//  int i = o;

//  int xx_start = get_local_id(0);
//  int xx_end = output_w;
//  int xx_step = get_local_size(0);

//  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);
//  int yy_end = output_h;
//  int yy_step = get_local_size(1)*get_num_groups(1);

//  // select input/output plane
//  gradOutput = gradOutput + o*output_w*output_h;
//  gradInput = gradInput + i*input_w*input_h;

//  // compute gradInput
//  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
//    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
//      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
//      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
//      float z = *ptr_gradOutput;
//      int kx, ky;
//      for(ky = 0; ky < kH; ky++) {
//        for(kx = 0; kx < kW; kx++) {
//          atomicAdd(&(ptr_gradInput[kx]), z / float(kW*kH));
//        }
//        ptr_gradInput += input_w;
//      }
//    }
//  }
//}

