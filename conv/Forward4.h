#pragma once

#include "conv/Forward.h"

class ClConvolver;
class CLKernel;
class AddBias;
class THClTensor;
class THClState;
class EasyCL;

#define VIRTUAL virtual
#define STATIC static

class Forward4 : public Forward {
public:
  THClState *state;
  int device;
  ClConvolver *conv;

  CLKernel *kernel;
  AddBias *addBias;

  int workgroupSize;
  int pixelsPerThread;

  // [[[cog
  // import cog_addheaders
  // cog_addheaders.addv2()
  // ]]]
  // generated, using cog:

  public:
  VIRTUAL ~Forward4();
  VIRTUAL void forward(THClState *state, THClTensor *input, THClTensor *weight, THClTensor *bias, THClTensor *output);
  Forward4(THClState *state, int device, ClConvolver *conv);

  // [[[end]]]
};

