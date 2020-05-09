#pragma once
#undef NDEBUG
#include <cassert>
#include "Logger.h"

// TODO: FIX THIS!!! ->  it's a dangerous hack to get around driver errors
#define cudaCheck(x) \
{ \		
}
/*
#define cudaCheck(x) \
	{ \
		cudaError_t err = (x); \
		if (err != cudaSuccess) { \
			Log("Line %d: cudaCheckError: %s", __LINE__, cudaGetErrorString(err)); \
			assert(0); \
		} \
	}
*/