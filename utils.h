#ifndef CLNN_UTILS_H
#define CLNN_UTILS_H

#ifdef __cplusplus 
extern "C" {
#endif

#include <lua.h>

#ifdef __cplusplus 
} // extern "C"
#endif

#include "THClGeneral.h"

THClState* getCltorchState(lua_State* L);

#endif

