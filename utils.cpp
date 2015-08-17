#include "utils.h"

THClState* getCltorchState(lua_State* L)
{
    lua_getglobal(L, "cltorch");
    lua_getfield(L, -1, "getState");
    lua_call(L, 0, 1);
    THClState *state = (THClState*) lua_touserdata(L, -1);
    lua_pop(L, 2);
    return state;
}

