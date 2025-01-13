#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_random.cuh"
#include "cuda_referencedeck.cuh"
#include "cuda_simrun.cuh"
#include "cuda_utils.cuh"

// Version 8:
//  - local counts are uint64_t to allow much larger simulations
//  - hand/cards represented by a uint64_t
//  - hand evaluation is adapted from v3
//  - shared memory for per-thread deck storage
//  - reference deck in shared memory as well
//  - the per-thread counts are also in shared memory

__device__ PokerHandType getHandType_v8(const uint8_t* hand, const CudaCard* refDeck) {
  uint64_t hv = refDeck[hand[0]].v + refDeck[hand[1]].v + refDeck[hand[2]].v + refDeck[hand[3]].v + refDeck[hand[4]].v;
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask1   = static_cast<uint64_t>(0b0000000000000001001001001000000000000000000000000000000000000000ULL);
  bool     isFlush = (0 < __popcll(hv & (hv >> 2) & mask1));
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask2        = static_cast<uint64_t>(0b0000000000000000000000000000000000000000000000000001001001001001ULL);
  bool     isStraight   = (5 == __popcll(hv & (mask2 << 24)));
  bool     isRoyalFlush = isFlush && isStraight;
  isStraight |= (5 == __popcll(hv & (mask2 << 21)));
  isStraight |= (5 == __popcll(hv & (mask2 << 18)));
  isStraight |= (5 == __popcll(hv & (mask2 << 15)));
  isStraight |= (5 == __popcll(hv & (mask2 << 12)));
  isStraight |= (5 == __popcll(hv & (mask2 << 9)));
  isStraight |= (5 == __popcll(hv & (mask2 << 6)));
  isStraight |= (5 == __popcll(hv & (mask2 << 3)));
  isStraight |= (5 == __popcll(hv & mask2));
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask3 = static_cast<uint64_t>(0b0000000000000000000000000001000000000000000000000000001001001001ULL);
  isStraight |= (5 == __popcll(hv & mask3));  // Ace low straight
  bool isStraightFlush = isStraight && isFlush;
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask4         = static_cast<uint64_t>(0b0000000000000000000000000001001001001001001001001001001001001001ULL);
  bool     isFourOfAKind = (1 == __popcll(hv & (mask4 << 2)));
  bool     isThreeOfAKind = (1 == __popcll((hv + mask4) & (mask4 << 2)));
  bool     twoGroups      = (2 == __popcll(hv & (mask4 << 1)));
  bool     oneGroup       = (1 == __popcll(hv & (mask4 << 1)));
  return isRoyalFlush      ? PokerHandType::RoyalFlush
         : isStraightFlush ? PokerHandType::StraightFlush
         : isFlush         ? PokerHandType::Flush
         : isStraight      ? PokerHandType::Straight
         : isFourOfAKind   ? PokerHandType::FourOfAKind
         : isThreeOfAKind  ? (twoGroups ? PokerHandType::FullHouse : PokerHandType::ThreeOfAKind)
         : twoGroups       ? PokerHandType::TwoPair
         : oneGroup        ? PokerHandType::OnePair
                           : PokerHandType::HighCard;
}

__global__ void simKernel_v8c(curandState* states, uint64_t* globalCounts, uint64_t handsPerThread) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t shared_mem[];
  CudaCard*                 refDeck     = (CudaCard*)shared_mem;
  uint8_t*                  myDeck      = &shared_mem[52 * 8 + threadIdx.x * (56 + 10 * 8)];
  uint64_t*                 localCounts = (uint64_t*)&myDeck[56];

  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < 52; ++i) refDeck[i] = getCudaCard64(i);  // Initialize reference deck
  for (int i = 0; i < 10; ++i) localCounts[i] = 0;
  __syncthreads();

  curandState localState = states[tid];
#pragma unroll
  for (int i = 0; i < 52; ++i) myDeck[i] = i;  // Initialize local deck with cards 0..51

  for (uint64_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, myDeck);

#pragma unroll
    for (int j = 0; j < 48; ++j) {
      PokerHandType type = getHandType_v8(myDeck + j, refDeck);
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) atomicAdd(&globalCounts[i], localCounts[i]);

  states[tid] = localState;
}

__global__ void simKernel_v8x(xoro_state128* states, uint64_t* globalCounts, uint64_t handsPerThread) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t shared_mem[];
  CudaCard*                 refDeck     = (CudaCard*)shared_mem;
  uint8_t*                  myDeck      = &shared_mem[52 * 8 + threadIdx.x * (56 + 10 * 8)];
  uint64_t*                 localCounts = (uint64_t*)&myDeck[56];

  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < 52; ++i) refDeck[i] = getCudaCard64(i);  // Initialize reference deck
  __syncthreads();

  xoro_state128 localState = states[tid];
#pragma unroll
  for (int i = 0; i < 52; ++i) myDeck[i] = i;  // Initialize local deck with cards 0..51
#pragma unroll
  for (int i = 0; i < 10; ++i) localCounts[i] = 0;

  for (uint64_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, myDeck);

#pragma unroll
    for (int j = 0; j < 48; ++j) {
      PokerHandType type = getHandType_v8(myDeck + j, refDeck);
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) atomicAdd(&globalCounts[i], localCounts[i]);

  states[tid] = localState;
}

template <typename F> DeviceConfig getDeviceConfig_v8(int maxBlockSize, F func) {
  DeviceConfig config;
  int          device;
  CHECK_CUDA_ERROR(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

  // Calculate shared memory requirements for the optimized kernel
  size_t sharedMemorySize =
      52 * sizeof(CudaCard)                                             // Reference deck in shared memory
      + maxBlockSize * (56 * sizeof(uint8_t) + 10 * sizeof(uint64_t));  // Local deck for each thread

  // Ensure we don't exceed device shared memory limits
  if (sharedMemorySize > prop.sharedMemPerBlock) {
    fprintf(stderr,
            "Warning: Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n",
            sharedMemorySize,
            prop.sharedMemPerBlock);
    // Adjust block size to fit shared memory
    maxBlockSize = prop.sharedMemPerBlock / ((52 * sizeof(CudaCard)) + (56 * sizeof(uint8_t)) + 10 * sizeof(uint64_t));
    maxBlockSize = (maxBlockSize / prop.warpSize) * prop.warpSize;  // Round to nearest warp size
    // Recalculate shared memory size
    sharedMemorySize = 52 * sizeof(CudaCard) + maxBlockSize * (56 * sizeof(uint8_t) + 10 * sizeof(uint64_t));
  }

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

void simRunCUDA_v8(SimulationResult& result) {
  CUDA_DEBUG("Initializing CUDA calculation");

  // Initialize the reference deck in constant memory
  initializeReferenceDeck64();

  DeviceConfig config;
  if (result.cudaRngMethod == RNGMethod::CURAND) {
    config = getDeviceConfig_v8(result.cudaBlockSize, &simKernel_v8c);
  } else {
    config = getDeviceConfig_v8(result.cudaBlockSize, &simKernel_v8x);
  }

  const uint64_t numThreads     = config.blockSize * config.numBlocks;
  const uint64_t handsPerThread = (result.numHands + numThreads - 1) / numThreads;
#if CUDA_DEBUG_ENABLED
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  //size_t alignedSize = (sizeof(xoro_state128) + 7) & ~7;
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

  size_t sharedMemSize = 52 * sizeof(CudaCard) + config.blockSize * (56 * sizeof(uint8_t) + 10 * sizeof(uint64_t));
  dim3   grid(config.numBlocks), block(config.blockSize);
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
      simKernel_v8c<<<grid, block, sharedMemSize>>>(d_curandStates, d_counts, handsPerThread);
    } else {
      simKernel_v8x<<<grid, block, sharedMemSize>>>(d_xoroStates, d_counts, handsPerThread);
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