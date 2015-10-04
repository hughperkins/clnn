#pragma once

class THClState;
class THClTensor;

class BackGrad {
public:
  virtual ~BackGrad() {}
  virtual void updateGradInput(THClState *state, THClTensor *input, THClTensor *gradOutput, THClTensor *weight, THClTensor *gradInput) = 0;
};

