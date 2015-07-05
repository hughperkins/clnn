// from SpatialMaxPooling.cu:

/*
 * Description:
 *    this function maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
kernel void maxpool(const global float *input_data, int input_offset,
    global float *output_data, int output_offset, 
    global float *indices_data, int indices_offset,
    int indices_x_offset,
    int indices_y_offset,
    int input_n, int input_h, int input_w,
    int kH, int kW, int dH, int dW)
{
  global const float *input = input_data + input_offset;
  global float *output = output_data + output_offset;
  global float *indices_x = indices_data + indices_offset + indices_x_offset;
  global float *indices_y = indices_data + indices_offset + indices_y_offset;

  // iterators
  int xx, yy;

  // output size
  const int output_w = floor((float)(input_w - kW) / dW + 1);
  const int output_h = floor((float)(input_h - kH) / dH + 1);

  // compute offsets based on thread/block ID
  int o = get_group_id(0);
  int i = o;
  //int k = get_group_id(0) % input_n;

  int xx_start = get_local_id(0);
  int xx_end = output_w;
  const int xx_step = get_local_size(0);

  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);
  int yy_end = output_h;
  const int yy_step = get_local_size(1)*get_num_groups(1);

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      // Compute the mean of the input image...
      global const float *ptr_input = input + yy*dH*input_w + xx*dW;
      global float *ptr_output = output + yy*output_w + xx;
      global float *ptr_ind_x = indices_x + yy*output_w + xx;
      global float *ptr_ind_y = indices_y + yy*output_w + xx;
      int argmax_x = -1;
      int argmax_y = -1;
      float max = -FLT_MAX;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          float val = ptr_input[kx];
          if (val > max) {
            max = val;
            argmax_x = kx;
            argmax_y = ky;
          }
        }
        ptr_input += input_w; // next input line
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind_x = argmax_x + 1;
      *ptr_ind_y = argmax_y + 1;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
kernel void maxgradinput(global float *gradInput_data, int gradInput_offset,
    global const float *gradOutput_data, int gradOutput_offset,
    global const float *indices_data, int indices_offset,
    int indices_x_offset, int indices_y_offset,
   int input_n, int input_h, int input_w,
   int kH, int kW, int dH, int dW)
{
  global float *gradInput = gradInput_data + gradInput_offset;
  global const float *gradOutput = gradOutput_data + gradOutput_offset;
  global const float *indices_x = indices_data + indices_offset + indices_x_offset;
  global const float *indices_y = indices_data + indices_offset + indices_y_offset;

  // iterators
  int xx, yy;

  // output size
  const int output_w = floor((float)(input_w - kW) / dW + 1);
  const int output_h = floor((float)(input_h - kH) / dH + 1);

  // compute offsets based on thread/block ID
  int o = get_group_id(0);
  int i = o;
  //int k = get_group_id(0) % input_n;

  int xx_start = get_local_id(0);
  int xx_end = output_w;
  int xx_step = get_local_size(0);

  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);
  int yy_end = output_h;
  int yy_step = get_local_size(1)*get_num_groups(1);

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      global float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      global const float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      global const float *ptr_ind_x = indices_x + yy*output_w + xx;
      global const float *ptr_ind_y = indices_y + yy*output_w + xx;
      float z = *ptr_gradOutput;

      int argmax_x = (*ptr_ind_x)-1;
      int argmax_y = (*ptr_ind_y)-1;

      ptr_gradInput[argmax_x + argmax_y*input_w] += z;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 *    when kH != dH or kW != dW (uses atomic add)
 */
//kernel void atomicmaxgradinput(
//  global float *gradInput_data, int gradInput_offset,
//  global float *gradOutput_data, int gradOutput_offset,
//  global float *indices_data, int indices_offset,
//  int indices_x_offset, int indices_y_offset,
//  int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW
//)
//{
//  global float *gradInput = gradInput_data + gradInput_offset;
//  global float *gradOutput = gradOutput_data + gradOutput_offset;
//  global float *indices_x = indices_data + indices_offset + indices_x_offset;
//  global float *indices_y = indices_data + indices_offset + indices_y_offset;

//  // iterators
//  int xx, yy;

//  // output size
//  int output_w = (input_w - kW) / dW + 1;
//  int output_h = (input_h - kH) / dH + 1;

//  // compute offsets based on thread/block ID
//  int o = get_group_id(0);
//  int i = o;
//  //int k = get_group_id(0) % input_n;

//  int xx_start = get_local_id(0);
//  int xx_end = output_w;
//  int xx_step = get_local_size(0);

//  int yy_start = get_local_size(1)*get_group_id(1) + get_local_id(1);
//  int yy_end = output_h;
//  int yy_step = get_local_size(1)*get_num_groups(1);

//  // select input/output plane
//  gradOutput = gradOutput + o*output_w*output_h;
//  gradInput = gradInput + i*input_w*input_h;
//  indices_x = indices_x + o*output_w*output_h;
//  indices_y = indices_y + o*output_w*output_h;

//  // compute gradInput
//  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
//    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
//      global float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
//      global float *ptr_gradOutput = gradOutput + yy*output_w + xx;
//      global float *ptr_ind_x = indices_x + yy*output_w + xx;
//      global float *ptr_ind_y = indices_y + yy*output_w + xx;
//      float z = *ptr_gradOutput;

//      int argmax_x = (*ptr_ind_x)-1;
//      int argmax_y = (*ptr_ind_y)-1;

//      // atomic add since different threads could update same variable
////      atomicAdd(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);
//      // hmmm, this doesnt work with float :-(  need another way...
//      atomic_add(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);
//    }
//  }
//}

