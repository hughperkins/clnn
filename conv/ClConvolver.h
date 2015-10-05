#pragma once

extern "C" {
  #include "lua.h"
  void clnn_ClConvolver_init(lua_State *L);
}

class Forward;
class BackGrad;
class GradWeights;

int ClConvolver_new(lua_State *L);

class ClConvolver {
public:
  int refCount;

  int batch;
  int batchSize;

  int dH;
  int dW;
  int kH;
  int kW;
  int nInputPlane;
  int nOutputPlane;
  int padW;
  int padH;

  int inputWidth;
  int inputHeight;
  int outputWidth;
  int outputHeight;

  Forward *forwarder;
  BackGrad *backGradder;
  GradWeights *gradWeights;

  ClConvolver() {
  }
  ~ClConvolver() {
  }
};

