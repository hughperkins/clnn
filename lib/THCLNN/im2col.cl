// from SpatialConvolutionMM.cu:

// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
kernel void im2col_kernel(const int n, const global float* im_data, int im_offset,
    const int inW, const int inH,
     const int kW, const int kH, 
    const int dW, const int dH,
    const int padW, const int padH,
    const int outW, const int outH,
    global float* col_data, int col_offset) {
  global const float *data_im = im_data + im_offset;
  global float *data_col = col_data + col_offset;

  CL_KERNEL_LOOP(index, n) {
    int w_out = index % outW;
    index /= outW;
    int h_out = index % outH;
    int channel_in = index / outH;
    int channel_out = channel_in * kW * kH;
    int w_in = w_out * dW - padW;
    int h_in = h_out * dH - padH;
    data_col += (channel_out * outH + h_out) * outW + w_out;
    data_im += (channel_in * inH + h_in) * inW + w_in;
    for (int i = 0; i < kH; ++i) {
      for (int j = 0; j < kW; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < inH && w < inW) ?
          data_im[i * inW + j] : 0;
        data_col += outH * outW;
      }
    }
  }
}

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
// adapted for use with groups, as described in
// "OpenCL caffe: Accelerating and enabling a cross platform machine" by Junli Gu et al
// imageIndex will control where in columns, the image is copied, and columns will have in fact
// numimages *
kernel void im2col_kernel_batched(const int n, const global float* im_data, int im_offset,
    const int inW, const int inH,
     const int kW, const int kH, 
    const int dW, const int dH,
    const int padW, const int padH,
    const int outW, const int outH,
    const int numImages, const int imageIdx,
    global float* col_data, int col_offset) {
  global const float *data_im = im_data + im_offset;
  global float *data_col = col_data + col_offset + imageIdx * outH * outW;

  CL_KERNEL_LOOP(index, n) {
    int w_out = index % outW;
    index /= outW;
    int h_out = index % outH;
    int channel_in = index / outH;
    int channel_out = channel_in * kW * kH;
    int w_in = w_out * dW - padW;
    int h_in = h_out * dH - padH;
    data_col += (channel_out * outH + h_out) * outW + w_out;
    data_im += (channel_in * inH + h_in) * inW + w_in;
    for (int i = 0; i < kH; ++i) {
      for (int j = 0; j < kW; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < inH && w < inW) ?
          data_im[i * inW + j] : 0;
        data_col += outH * outW * numImages;
      }
    }
  }
}
kernel void col2im_kernel(const int n, global const float* col_data, int col_offset,
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    const int outW, const int outH,
    global float* im_data, int im_offset) {
  global float *data_im = im_data + im_offset;
  global const float *data_col = col_data + col_offset;

  CL_KERNEL_LOOP(index, n) {
    float val = 0;
    int w = index % inW + padW;
    int h = (index / inW) % inH + padH;
    int c = index / (inW * inH);
    // compute the start and end of the output
    int w_col_start = (w < kW) ? 0 : (w - kW) / dW + 1;
    int w_col_end = min(w / dW + 1, outW);
    int h_col_start = (h < kH) ? 0 : (h - kH) / dH + 1;
    int h_col_end = min(h / dH + 1, outH);
    /*
       for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
    // the col location: [c * width * height + h_out, w_out]
    int c_col = c * patch_h * patch_w + (h - h_col * dH) * ksize + (w - w_col * dW);
    val += data_col[(c_col * height_col + h_col) * width_col + w_col];
    }
    }
     */
    // equivalent implementation
    int offset = (c * kH * kW + h * kW + w) * outH * outW;
    int coeff_h_col = (1 - dH * kW * outH) * outW;
    int coeff_w_col = (1 - dW * outH * outW);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

