#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_random.cuh"
#include "cuda_simrun.cuh"
#include "cuda_utils.cuh"

// Cards are represented as 8-bit integers 0..51
// uint8_t rank = card >> 2;
// uint8_t suit = card & 0x3;

__device__ PokerHandType getPokerHandType_v1(const uint8_t* hand) {
  uint8_t suits[5] = {static_cast<uint8_t>(hand[0] & 0x3),
                      static_cast<uint8_t>(hand[1] & 0x3),
                      static_cast<uint8_t>(hand[2] & 0x3),
                      static_cast<uint8_t>(hand[3] & 0x3),
                      static_cast<uint8_t>(hand[4] & 0x3)};
  uint8_t ranks[5] = {0};
  // Sort ranks in ascending order
  ranks[0] = static_cast<uint8_t>(hand[0] >> 2);
#pragma unroll
  for (uint8_t i = 1; i < 5; ++i) {
    uint8_t newRank = static_cast<uint8_t>(hand[i] >> 2);
#pragma unroll
    for (uint8_t j = i; j > 0; --j) {
      ranks[j] = newRank > ranks[j - 1] ? newRank : ranks[j - 1];
      newRank  = newRank > ranks[j - 1] ? ranks[j - 1] : newRank;
    }
    ranks[0] = newRank < ranks[0] ? newRank : ranks[0];
  }
  bool isFlush = ((suits[0] == suits[1]) && (suits[1] == suits[2]) && (suits[2] == suits[3]) && (suits[3] == suits[4]));
  bool isStraight = (((ranks[1] - ranks[0]) == 1) && ((ranks[2] - ranks[1]) == 1) && ((ranks[3] - ranks[2]) == 1) &&
                     (((ranks[4] - ranks[3]) == 1) || ((ranks[4] - ranks[3]) == 9)));
  if (isFlush)
    return isStraight ? (ranks[4] == 8) ? PokerHandType::RoyalFlush : PokerHandType::StraightFlush
                      : PokerHandType::Flush;
  if (isStraight) return PokerHandType::Straight;

  uint8_t numPair        = 0;
  bool    isThreeOfAKind = false;
  bool    isFourOfAKind  = false;

#pragma unroll
  for (int i = 0; i < 5; ++i) {
    int count = 1;
#pragma unroll
    for (int j = i + 1; j < 5; ++j) {
      if (ranks[i] == ranks[j]) {
        count++;
      }
    }
    if (count == 2) {
      numPair++;
    } else if (count == 3) {
      numPair--;
      isThreeOfAKind = true;
    } else if (count == 4) {
      isFourOfAKind = true;
    }
  }
  return isFourOfAKind    ? PokerHandType::FourOfAKind
         : isThreeOfAKind ? (numPair > 0) ? PokerHandType::FullHouse : PokerHandType::ThreeOfAKind
         : numPair == 2   ? PokerHandType::TwoPair
         : numPair == 1   ? PokerHandType::OnePair
                          : PokerHandType::HighCard;
}

__global__ void simKernel_v1c(curandState* states, uint64_t* globalCounts, uint32_t handsPerThread) {
  int         tid        = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = states[tid];
  uint8_t
           deck[52]        = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                              18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51};  // Local deck array for each thread
  uint32_t localCounts[10] = {0};

  for (uint32_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, deck);
#pragma unroll
    for (int j = 0; j < 48; ++j) {
      PokerHandType type = getPokerHandType_v1(deck + j);
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    atomicAdd(&globalCounts[i], localCounts[i]);
  }

  states[tid] = localState;
}

__global__ void simKernel_v1x(xoro_state128* states, uint64_t* globalCounts, uint32_t handsPerThread) {
  int           tid        = blockIdx.x * blockDim.x + threadIdx.x;
  xoro_state128 localState = states[tid];
  uint8_t
           deck[52]        = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                              18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51};  // Local deck array for each thread
  uint32_t localCounts[10] = {0};

  for (uint32_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, deck);
#pragma unroll
    for (int j = 0; j < 48; ++j) {
      PokerHandType type = getPokerHandType_v1(deck + j);
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    atomicAdd(&globalCounts[i], localCounts[i]);
  }

  states[tid] = localState;
}

template <typename F> DeviceConfig getDeviceConfig_v1(int maxBlockSize, F func) {
  DeviceConfig config;
  int          device;
  CHECK_CUDA_ERROR(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

  // Calculate shared memory requirements for the optimized kernel
  size_t sharedMemorySize = 0;

  // Optimize for maximum occupancy
  int minGridSize;
  int blockSize;
  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, sharedMemorySize));

  // Cap block size to max specified by user and ensure block size is multiple of warp size
  config.blockSize = (std::min(blockSize, maxBlockSize) / prop.warpSize) * prop.warpSize;

  // Calculate optimal number of blocks
  int blocksPerSM;
  CHECK_CUDA_ERROR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, func, config.blockSize, sharedMemorySize));

  config.numBlocks       = prop.multiProcessorCount * blocksPerSM;
  config.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
  config.numSMs          = prop.multiProcessorCount;

#if CUDA_DEBUG_ENABLED
  printf("Shared Memory Config:\n");
  printf("  Required: %zu bytes\n", sharedMemorySize);
  printf("  Available: %zu bytes\n", prop.sharedMemPerBlock);
  printf("  Recommended block Size: %d\n", blockSize);
  printf("  Actual block Size: %d\n", config.blockSize);
  printf("  Blocks per SM: %d\n", blocksPerSM);
#endif

  return config;
}

void simRunCUDA_v1(SimulationResult& result) {
  CUDA_DEBUG("Initializing CUDA calculation");

  DeviceConfig config;
  if (result.cudaRngMethod == RNGMethod::CURAND) {
    config = getDeviceConfig_v1(result.cudaBlockSize, &simKernel_v1c);
  } else {
    config = getDeviceConfig_v1(result.cudaBlockSize, &simKernel_v1x);
  }

  const uint64_t numThreads     = config.blockSize * config.numBlocks;
  const uint64_t handsPerThread = (result.numHands + numThreads - 1) / numThreads;
#if CUDA_DEBUG_ENABLED
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("\nGPU Properties:\n");
  printf("  Name: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  SMs: %d\n", prop.multiProcessorCount);
  printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Warp Size: %d\n", prop.warpSize);
  printf("  Max Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
  printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate * 1e-6);
  printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
  printf("GPU Config:\n");
  printf("  Block size: %d\n", config.blockSize);
  printf("  Num. blocks: %d\n", config.numBlocks);
  printf("  Num. threads: %llu\n", numThreads);
  printf("  Hands per thread: %llu\n", handsPerThread);
  //printf("  Memory alignment: %zu bytes\n", alignedSize);
  fflush(stdout);
#endif

  //size_t sharedMemSize = (10 * sizeof(uint64_t)) + (config.blockSize * 52 * sizeof(uint8_t));
  dim3           grid(config.numBlocks), block(config.blockSize);
  curandState*   d_curandStates = nullptr;
  xoro_state128* d_xoroStates   = nullptr;
  uint64_t*      d_counts       = nullptr;
  uint64_t*      h_counts       = new uint64_t[10]();

  try {
    CUDA_DEBUG("Allocating device memory");
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      CHECK_CUDA_ERROR(cudaMalloc(&d_curandStates, numThreads * sizeof(curandState)));
    } else {
      CHECK_CUDA_ERROR(cudaMalloc(&d_xoroStates, numThreads * sizeof(xoro_state128)));
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_counts, 10 * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_counts, 0, 10 * sizeof(uint64_t)));

    CUDA_DEBUG("Initializing RNG states");
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      initCurandRNG<<<grid, block>>>(d_curandStates, time(nullptr));
    } else {
      initXoroRNG<<<grid, block>>>(d_xoroStates, time(nullptr));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_DEBUG("Starting simulation");
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      simKernel_v1c<<<grid, block>>>(d_curandStates, d_counts, handsPerThread);
    } else {
      simKernel_v1x<<<grid, block>>>(d_xoroStates, d_counts, handsPerThread);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Add verification check
    uint64_t total = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(h_counts, d_counts, 10 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; ++i) {
      total += h_counts[i];
      result.counts[i] = h_counts[i];
    }
    result.numThreads    = numThreads;
    result.cudaBlockSize = config.blockSize;
    result.cudaNumBlocks = config.numBlocks;

#if CUDA_EXTENDED_DEBUG_ENABLED
    if (total == 0) {
      printf("Warning: No hands were counted! Check kernel execution.\n");
    } else {
      printf("Total hands processed: %llu\n", total);
    }
#endif
  } catch (const std::exception& e) {
    printf("Error during CUDA execution: %s\n", e.what());
    if (d_curandStates) cudaFree(d_curandStates);
    if (d_xoroStates) cudaFree(d_xoroStates);
    if (d_counts) cudaFree(d_counts);
    delete[] h_counts;
    throw;
  }

  // Cleanup
  if (d_curandStates) cudaFree(d_curandStates);
  if (d_xoroStates) cudaFree(d_xoroStates);
  if (d_counts) cudaFree(d_counts);
  delete[] h_counts;

  CUDA_DEBUG("Calculation complete");
}
