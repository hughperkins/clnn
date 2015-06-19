#!/bin/bash

source ~/torch/activate

# rm -Rf build

luarocks make rocks/clnn-scm-1.rockspec || exit 1

export LUA_PATH="$LUA_PATH;${PWD}/thirdparty/?.lua"

if [[ ! -v LUAEXE ]]; then {
     LUAEXE=luajit
   # LUAEXE=lua
} fi
echo using luaexe: ${LUAEXE}

if [[ x${RUNGDB} == x1 ]]; then {
  rungdb.sh ${LUAEXE} test/test-layers.lua
} else {
  ${LUAEXE} test/test-layers.lua
} fi

