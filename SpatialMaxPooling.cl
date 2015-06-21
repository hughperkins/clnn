// from SpatialMaxPooling.cu:

/*
 * Description:
 *    this function maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
kernel void maxpool(float *input, float *output, float *indices_x, float *indices_y,
                        int input_n, int input_h, int input_w,
                        int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  const int output_w = (input_w - kW) / dW + 1;
  const int output_h = (input_h - kH) / dH + 1;

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
      float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_output = output + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
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
kernel void maxgradinput(float *gradInput, float *gradOutput, float *indices_x, float *indices_y,
                             int input_n, int input_h, int input_w,
                             int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

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
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
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
kernel void atomicmaxgradinput(
  float *gradInput, float *gradOutput, float *indices_x, float *indices_y,
  int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW
)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

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
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      float z = *ptr_gradOutput;

      int argmax_x = (*ptr_ind_x)-1;
      int argmax_y = (*ptr_ind_y)-1;

      // atomic add since different threads could update same variable
      atomicAdd(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);
    }
  }
}

