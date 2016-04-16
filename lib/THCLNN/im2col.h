#pragma once

void im2col(THClState *state, THClTensor* im, const int channels,
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    THClTensor* col);

void col2im(THClState *state, THClTensor* col,
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    THClTensor* im);

