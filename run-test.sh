#!/bin/bash

source ~/torch/activate

# rm -Rf build

luarocks make rocks/clnn-scm-1.rockspec

luajit test/test-clnn.lua

