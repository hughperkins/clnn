#pragma once

extern "C" {
  #include "lua.h"
  void clnn_ClConvolver_init(lua_State *L);
}

class Forward;
class BackGrad;

int ClConvolver_new(lua_State *L);

class ClConvolver {
public:
  int refCount;

  int batch;
  int batchSize;

  int dW;
  int dH;
  int kW;
  int kH;
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

  ClConvolver() {
  }
  ~ClConvolver() {
  }
};

