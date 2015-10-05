// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

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
  STATIC Forward *instance(THClState *state, int device, ClConvolver *conv);
  STATIC Forward *instanceSpecific(int idx, THClState *state, int device, ClConvolver *conv);

  // [[[end]]]
};

