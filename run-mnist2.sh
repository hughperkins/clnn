#!/bin/bash

source ~/torch/activate

# rm -Rf build

luarocks make rocks/clnn-scm-1.rockspec || exit 1

if [[ ! -v LUAEXE ]]; then {
     LUAEXE=luajit
    # LUAEXE=lua
    #LUAEXE=/norep/Downloads/lua-5.1.5/src/lua
} fi
echo using luaexe: ${LUAEXE}

if [[ x${RUNGDB} == x1 ]]; then {
  rungdb.sh ${LUAEXE} test/test-mnist2.lua
} else {
  ${LUAEXE} test/test-mnist2.lua
} fi

