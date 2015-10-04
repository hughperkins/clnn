#pragma once

#include "conv/Forward.h"

class ClConvolver;

class ForwardIm2Col : public Forward {
public:
  THClState *state;
  int device;
  ClConvolver *conv;

  THClTensor *columns;
  THClTensor *ones;
  
  ForwardIm2Col(THClState *state, int device, ClConvolver *conv);
  virtual ~ForwardIm2Col();
  virtual void forward(THClState *state, int batch, THClTensor *input, THClTensor *weight, THClTensor *bias, THClTensor *output);
};

