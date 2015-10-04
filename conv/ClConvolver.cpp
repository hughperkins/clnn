#include "conv/ClConvolver.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}

#include "THAtomic.h"
#include "THClGeneral.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
using namespace std;

static void ClConvolver_rawInit(ClConvolver *self) {
  self->refCount = 1;
}

int ClConvolver_new(lua_State *L) {
  ClConvolver *self = (ClConvolver*)THAlloc(sizeof(ClConvolver));
  self = new(self) ClConvolver();
  ClConvolver_rawInit(self);
  luaT_pushudata(L, self, "nn.ClConvolver");
  return 1;
}
static int ClConvolver_free(lua_State *L) {
  ClConvolver *self = (ClConvolver*)THAlloc(sizeof(ClConvolver));
  if(!self) {
    return 0;
  }
  if(THAtomicDecrementRef(&self->refCount))
  {
    self->~ClConvolver();
    THFree(self);
  }
  return 0;
}
static int ClConvolver_factory(lua_State *L) {
  THError("ClConvolver_factory not implemented");
  return 0;
}
static int ClConvolver_print(lua_State *L) {
  ClConvolver *self = (ClConvolver *)luaT_checkudata(L, 1, "nn.ClConvolver");
  cout << "ClConvolver " << self->nInputPlane << "->" << self->nOutputPlane << " " << self->kH << "x" << self->kW << endl;
  return 0;
}
static const struct luaL_Reg ClConvolver_funcs [] = {
  {"print", ClConvolver_print},
//  {"getRenderedKernel", ClConvolver_getRenderedKernel},
//  {"getRawKernel", ClConvolver_getRawKernel},
//  {"run", ClConvolver_run},
  {0,0}
};
void clnn_ClConvolver_init(lua_State *L)
{
  luaT_newmetatable(L, "nn.ClConvolver", NULL,
                    ClConvolver_new, ClConvolver_free, ClConvolver_factory);
  luaT_setfuncs(L, ClConvolver_funcs, 0);
  lua_pop(L, 1);
}

