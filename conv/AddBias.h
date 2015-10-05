// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "EasyCL.h"

class EasyCL;
class THClTensor;
class THClState;
class CLKernel;
class ClConvolver;

#define VIRTUAL virtual
#define STATIC static

// adds bias, during forward propagation, after convolutional kernel has run
// but before activation etc
class AddBias {
public:
  THClState *state;
  int device;
  ClConvolver *conv;

//  EasyCL *cl; // NOT delete
  CLKernel *kernel; // NOT delete

  // [[[cog
  // import cog_addheaders
  // cog_addheaders.addv2()
  // ]]]
  // generated, using cog:

  public:
  VIRTUAL ~AddBias();
  VIRTUAL void forward(
  int batchSize, int numFilters, int outputHeight, int outputWidth,
  THClTensor *output,
  THClTensor *bias
  );
  AddBias(THClState *state, int device, ClConvolver *conv);

  // [[[end]]]
};

