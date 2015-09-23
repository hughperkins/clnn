#pragma once

struct THClState;

int GET_CL_NUM_THREADS(THClState *state);

// CL: number of blocks for threads.
int GET_BLOCKS(THClState *state, const int N);

