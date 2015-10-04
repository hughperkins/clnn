#pragma once

#include "conv/BackGrad.h"

class ClConvolver;

class BackGradIm2Col : public BackGrad {
public:
  THClState *state;
  int device;
  ClConvolver *conv;

  THClTensor *gradColumns;

  BackGradIm2Col(THClState *state, int device, ClConvolver *conv);
  virtual ~BackGradIm2Col();
  virtual void updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *weight, THClTensor *gradInput);
};

