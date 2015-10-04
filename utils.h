#pragma once 

struct THClState;

#ifdef __cplusplus 
extern "C" {
#endif

#include <lua.h>
struct THClState* getCltorchState(lua_State* L);

#ifdef __cplusplus 
} // extern "C"
#endif

// #include "THClGeneral.h"

