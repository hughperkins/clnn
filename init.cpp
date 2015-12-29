#include <stdio.h>
#include <iostream>
#include <stdexcept>
using namespace std;

extern "C" {
  #include "lua.h"
//  #include "lauxlib.h"
  #include "utils.h"
//  #include "luaT.h"
}

#include "commit_generated.h"

extern "C" {
    int luaopen_libclnn( lua_State *L );
}

void clnn_ELU_init(lua_State *L);
void clnn_SpatialConvolutionMM_init(lua_State *L);
void clnn_SpatialMaxPooling_init(lua_State *L);
void clnn_SpatialAveragePooling_init(lua_State *L);
void clnn_SoftMax_init(lua_State *L);

static int clnn_about(lua_State *L)
{
  cout << "clnn.  OpenCL backend for Torch nn" << endl;
  cout << "Built from commit " << commit << endl;
  cout << "More info, doc: https://github.com/hughperkins/clnn" << endl;
  cout << "Issues: https://github.com/hughperkins/clnn/issues" << endl;
  return 0;
}

static const struct luaL_Reg clnn_stuff__ [] = {
  {"about", clnn_about},
  {NULL, NULL}
};

int luaopen_libclnn( lua_State *L ) {
  try {
    lua_newtable(L);
    luaL_setfuncs(L, clnn_stuff__, 0);

  //    printf("luaopen_libclnn called :-)\n");
    clnn_ELU_init(L);
    clnn_SpatialConvolutionMM_init(L);
    clnn_SpatialMaxPooling_init(L);
    clnn_SpatialAveragePooling_init(L);
    clnn_SoftMax_init(L);
//    cout << " try cout" << endl;
  } catch(runtime_error &e) {
    THError("Something went wrong: %s", e.what());
  }
  return 1;
}

