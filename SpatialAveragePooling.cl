// from SpatialAveragePooling.cu:

#define Dtype {{Dtype}}
#define COUNT_INCLUDE_PAD {{COUNT_INCLUDE_PAD}}

#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

{% if forward then %}
kernel void AvePoolForward(const int nthreads,
    global const Dtype* const bottom_data_data, int bottom_data_offset,
    const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    global Dtype* const top_data_data, int top_data_offset
    ) {
  global const Dtype* const bottom_diff = bottom_diff_data + bottom_diff_offset;
  global Dtype* const top_diff = top_diff_data + top_diff_offset;

  // bake in later, once working and if this layer is shown to contribute highly to
  // slowness
  CL_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    if(COUNT_INCLUDE_PAD)
      top_data[index] = aveval / ((hend - hstart) * (wend - wstart));
    else
      top_data[index] = aveval / pool_size;
  }
}
{% end %}

{% if backward then %}
kernel void AvePoolBackward(
    const int nthreads,
    global const Dtype* const top_diff_data, int top_diff_offset,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff_data, int bottom_diff_offset
    ) {
  global const Dtype * const top_diff = top_diff_data + top_diff_offset;
  global Dtype *const bottom_diff = bottom_diff_data + bottom_diff_offset;
  CL_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        if(COUNT_INCLUDE_PAD)
          gradient += top_diff_slice[ph * pooled_width + pw] / ((hend - hstart) * (wend - wstart));
        else
          gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}
{% end %}
