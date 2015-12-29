#ifndef CLNN_UTILS_H
#define CLNN_UTILS_H

struct THClState;

#ifdef __cplusplus 
extern "C" {
#endif

#include <lua.h>
#include "luaT.h"
#include "TH.h"
struct THClState* getCltorchState(lua_State* L);

#if LUA_VERSION_NUM == 501
/*
** Adapted from Lua 5.2.0
*/
void luaL_setfuncs (lua_State *L, const luaL_Reg *l, int nup);
#endif

#ifdef __cplusplus 
} // extern "C"
#endif

#include "THClGeneral.h"


#endif

