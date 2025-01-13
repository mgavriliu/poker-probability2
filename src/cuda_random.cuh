#pragma once

#include <curand_kernel.h>
#include <stdint.h>
#include "random.h"

struct xoro_state128 {
  uint64_t s[2];
};

__device__ inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

__device__ inline uint64_t xoro_next(xoro_state128* state) {
  const uint64_t s0 = state->s[0];
  uint64_t s1 = state->s[1];
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  state->s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
  state->s[1] = rotl(s1, 37);

  return result;
}

__device__ inline void xoro_init(xoro_state128* state, uint64_t seed) {
  uint64_t z = (seed + 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  state->s[0] = z ^ (z >> 31);

  z = (seed + 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  state->s[1] = z ^ (z >> 31);
}

__device__ inline void shuffleDeck(curandState* state, uint8_t* deck) {
// Fisher-Yates shuffle
#pragma unroll
  for (uint8_t i = 51; i > 0; --i) {
    uint8_t j = curand(state) % (i + 1);
    uint8_t temp = deck[i];
    deck[i] = deck[j];
    deck[j] = temp;
  }
}

__device__ inline void shuffleDeck2(xoro_state128* state, uint8_t* deck) {
// Fisher-Yates shuffle using xoroshiro
#pragma unroll
  for (uint8_t i = 51; i > 0; --i) {
    uint8_t j = xoro_next(state) % (i + 1);
    uint8_t temp = deck[i];
    deck[i] = deck[j];
    deck[j] = temp;
  }
}

__device__ inline void shuffleDeck(xoro_state128* state, uint8_t* deck) {
  if (!deck) return;

#pragma unroll
  for (int i = 51; i > 0; i -= 8) {
    uint64_t rand = xoro_next(state);

    // Since rand is 64 bits, and we only need an 8 bit random number
    // for each iteration of the loop. We can use all 8 random bytes
    // available in the 64 bit random number before asking for a new one.
#pragma unroll
    for (int byte = 0; byte < 8 && i - byte >= 0; ++byte) {
      // Extract each byte and use it to generate an index
      uint8_t randByte = (rand >> (byte * 8)) & 0xFF;
      // Map the byte to a valid index range (0 to i)
      int j = randByte % (i - byte + 1);

      // Swap cards
      uint8_t temp = deck[i - byte];
      deck[i - byte] = deck[j];
      deck[j] = temp;
    }
  }
}

__global__ void initXoroRNG(xoro_state128* states, unsigned long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  xoro_init(&states[tid], seed + tid);
}

__global__ void initCurandRNG(curandState* states, unsigned long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed + tid, 0, 0, &states[tid]);
}