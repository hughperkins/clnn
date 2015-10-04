#pragma once

class THClTensor;
class THClState;

class Forward {
public:
  virtual ~Forward() {}
  virtual void forward(THClState *state, THClTensor *input, THClTensor *weight, THClTensor *bias, THClTensor *output) = 0;
};

