#include "common.h"
#include "utils.h"
#include "DeviceInfo.h"

int GET_CL_NUM_THREADS(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

int GET_BLOCKS(THClState *state, const int N) {
  return (N + GET_CL_NUM_THREADS(state) - 1) / GET_CL_NUM_THREADS(state);
}

