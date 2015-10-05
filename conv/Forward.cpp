#include "conv/Forward.h"
#include "conv/Forward4.h"
#include "conv/ForwardIm2Col.h"
#include "THGeneral.h"

#include <iostream>
#include <string>
#include <sstream>
using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 
  
STATIC int Forward::getNumImplementations() {
  return 2;
}
STATIC bool Forward::plausiblyOptimal(int index, ClConvolver *conv) {
  return true;
}
STATIC Forward *Forward::instance(THClState *state, int device, ClConvolver *conv) {
    return new Forward4(state, device, conv);
}
STATIC Forward *Forward::instanceSpecific(int idx, THClState *state, int device, ClConvolver *conv) {
  if(idx == 0) {
    return new ForwardIm2Col(state, device, conv);
  } else if(idx == 1) {
    return new Forward4(state, device, conv);
  } else {
    THError( __FILE__ ":%i Forward::instanceSpecific: no instance defined for index %i ", __LINE__, idx);
    return 0;
  }
}

