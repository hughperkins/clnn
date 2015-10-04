#pragma once

class THClTensor;
class THClState;
class ClConvolver;

#define STATIC static
#define VIRTUAL virtual

class Forward {
public:
  virtual ~Forward() {}
  virtual void forward(THClState *state, THClTensor *input, THClTensor *weight, THClTensor *bias, THClTensor *output) = 0;

  // [[[cog
  // import cog_addheaders
  // cog_addheaders.add()
  // ]]]
  // generated, using cog:
  STATIC int getNumImplementations();
  STATIC bool plausiblyOptimal(int index, ClConvolver *conv);
  STATIC Forward *instanceSpecific(int idx, THClState *state, int device, ClConvolver *conv);

  // [[[end]]]
};

