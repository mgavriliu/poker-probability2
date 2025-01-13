#pragma once

#include <stdexcept>  // Include for std::runtime_error

#define CUDA_DEBUG_ENABLED 0
#define CUDA_EXTENDED_DEBUG_ENABLED 0

template <typename T> void check_cuda(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err),
            cudaGetErrorString(err), func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)

#if CUDA_EXTENDED_DEBUG_ENABLED
#define CUDA_DEBUG(x)                                     \
  do {                                                    \
    printf("DEBUG [%s:%d]: %s\n", __FILE__, __LINE__, x); \
    fflush(stdout);                                       \
  } while (0)
#else
#define CUDA_DEBUG(x)
#endif

static void throwRuntimeError(const char* message) { throw std::runtime_error(message); }