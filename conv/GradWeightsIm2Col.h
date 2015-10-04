#pragma once

#include "conv/GradWeights.h"

class ClConvolver;

class GradWeightsIm2Col : public GradWeights {
public:
  THClState *state;
  int device;
  ClConvolver *conv;

  THClTensor *columns;
  THClTensor *ones;
  
  GradWeightsIm2Col(THClState *state, int device, ClConvolver *conv);
  virtual ~GradWeightsIm2Col();
  virtual void accGradParameters(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradWeight, THClTensor *gradBias, float scale);
};

