#pragma once

class THClState;
class THClTensor;

class GradWeights {
public:
  virtual ~GradWeights() {}
  virtual void accGradParameters(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *gradWeight, THClTensor *gradBias, float scale) = 0;
};

