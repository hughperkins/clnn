cmake_minimum_required( VERSION 2.8.12 )

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -save-temps")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")


Find_package(Lua51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THCl")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
message("Torch_INSTALL_LIB ${Torch_INSTALL_LIB}")

#add_library( clnn SHARED init.cpp )

set(src init.cpp MSECriterion.cpp utils.c)
set(luasrc init.lua)
ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}" )

TARGET_LINK_LIBRARIES(clnn luaT THCl TH)


