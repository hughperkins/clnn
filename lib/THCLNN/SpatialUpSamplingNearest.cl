// from SpatialUpSamplingNearest.cu:

/*__device__*/ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;

}
/*__device__*/ int translate_idx_inv(int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;

}

kernel void upscale(global float *input_data, int input_offset, global float *output_data, int output_offset, int no_elements,
                        int scale_factor, int d1, int d2, int d3)
{
  global float *input = input_data + input_offset;
  global float *output = output_data + output_offset;
  // output offset:
  long ii = get_local_id(0) + get_local_size(0) * get_group_id(0);
  ii += get_local_id(1) + get_local_size(1) * (get_local_size(0) * get_num_groups(0)) * get_group_id(1);
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}

/*
 * Description:
 */
kernel void downscale(global float *gradInput_data, int gradInput_offset, global float *gradOutput_data, int gradOutput_offset, int no_elements,
                              int scale_factor, int d1, int d2, int d3)
{
  global float *gradInput = gradInput_data + gradInput_offset;
  global float *gradOutput = gradOutput_data + gradOutput_offset;
  // output offset:
  long ii = get_local_id(0) + get_local_size(0) * get_group_id(0);
  ii += get_local_id(1) + get_local_size(1) * (get_local_size(0) * get_num_groups(0)) * get_group_id(1);
  if (ii >= no_elements) return;
  for (int i=0; i < scale_factor; i++){
    for(int j=0; j < scale_factor; j++){
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput[ii] += gradOutput[ipidx];
    }
  }
}
