CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch REQUIRED)
ENDIF()

IF(NOT THCLNN_INSTALL_LIB_SUBDIR)
  SET(THCLNN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "THCLNN install library directory")
ENDIF()

#FILE(GLOB src-cl *.cpp)
set(src-cl Abs.cpp SpatialConvolutionMM.cpp SpatialFullConvolution.cpp ELU.cpp SpatialAveragePooling.cpp SpatialMaxPooling.cpp
  SoftMax.cpp Tanh.cpp common.cpp SpatialUpSamplingNearest.cpp im2col.cpp
)

add_library(THCLNN MODULE ${src-cl})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(THCLNN THCl TH EasyCL)

INSTALL(TARGETS THCLNN
  LIBRARY DESTINATION ${THCLNN_INSTALL_LIB_SUBDIR})
