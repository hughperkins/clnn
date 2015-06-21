cmake_minimum_required( VERSION 2.8.12 )

OPTION(DEV_RUN_COG "cltorch maintainers only, otherwise set to 'OFF'." OFF)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -save-temps")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")


Find_package(Lua51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THCl")
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/easycl")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
message("Torch_INSTALL_LIB ${Torch_INSTALL_LIB}")

#add_library( clnn SHARED init.cpp )

set(src init.cpp MSECriterion.cpp utils.c SpatialConvolutionMM.cpp SpatialMaxPooling.cpp )
set(luasrc init.lua MSECriterion.lua)
ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}" )

if(DEV_RUN_COG)
    add_custom_target(
        cog_clnn
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies( clnn cog_clnn )
endif(DEV_RUN_COG)

TARGET_LINK_LIBRARIES(clnn luaT THCl TH)


