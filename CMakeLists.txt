cmake_minimum_required( VERSION 2.8.12 )

OPTION(DEV_RUN_COG "cltorch maintainers only, otherwise set to 'OFF'." OFF)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -save-temps")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")

execute_process( COMMAND git log -n 1 --pretty=%h OUTPUT_VARIABLE git_commit OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process( COMMAND echo string commit="${git_commit}" > ${CMAKE_CURRENT_SOURCE_DIR}/commit.h )
file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/clnn_commit_generated.h
  CONTENT "const char *clnn_commit=\"${git_commit}\";\n"
)

#Find_package(Lua51 REQUIRED)
#include_directories(${LUA_INCLUDE_DIR})

add_definitions(-DUSE_CLEW)

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THCl")
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/easycl")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
#message("Torch_INSTALL_LIB ${Torch_INSTALL_LIB}")

#add_library( clnn SHARED init.cpp )

set(src init.cpp utils.cpp
)

set(luasrc init.lua MSECriterion.lua Pointwise.lua Threshold.lua LookupTable.lua
  LogSoftMax.lua ClassNLLCriterion.lua StatefulTimer.lua THCLNN.lua SpatialUpSamplingNearest.lua
  Narrow.lua CMulTable.lua test.lua test/testSpatialMaxPooling.lua test/testSpatialConvolutionMM.lua
  test/testLookupTable.lua test/testMSECriterion.lua test/testSpatialUpSamplingNearest.lua
  test/testELU.lua test/testhelpers.lua
  test/testClassNLLCriterion.lua test/testSoftMax.lua test/testLogSoftMax.lua test/testSpatialAveragePooling.lua)

ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}" )

if(DEV_RUN_COG)
    add_custom_target(
        cog_clnn
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/lib/THCLNN/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/lib/THCLNN/*.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies( clnn cog_clnn )
endif(DEV_RUN_COG)

TARGET_LINK_LIBRARIES(clnn luaT THCl TH)

ADD_SUBDIRECTORY(lib)
