// from SpatialMaxPooling.cu:

// kernels borrowed from Caffe

#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

#define Dtype {{Dtype}}


{% if forward %}
kernel void MaxPoolForward(const int nthreads, const Dtype* bottom_data_data,
    int bottom_data_offset,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data_data,
    int top_data_offset,
    Dtype* top_mask_data, int top_mask_offset) {

  global Dtype *bottom_data = bottom_data_data + bottom_data_offset;
  global Dtype *top_data = top_data_data + top_data_offset;
  global Dtype *top_mask = top_mask_data + top_mask_offset;

  CL_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    top_mask[index] = maxidx + 1;
  }
}
{% end %}

{% if backward %}
kernel void MaxPoolBackward(const int nthreads, const Dtype* top_diff_data,
    int top_diff_offset,
    const Dtype* top_mask_data, const int top_mask_offset, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff_data, int bottom_diff_offset) {

  global Dtype *top_diff = top_diff_data + top_diff_offset;
  global Dtype *top_mask = top_mask_data + top_mask_offset;
  global Dtype *bottom_diff = bottom_diff_data + bottom_diff_offset;

  // looks like this could probably be baked, anyway, we can do that later...
  CL_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    top_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	if (top_mask[ph * pooled_width + pw] - 1 == h * width + w) {
	  gradient += top_diff[ph * pooled_width + pw];
	}
      }
    }
    bottom_diff[index] = gradient;
  }
}
{% end %}
