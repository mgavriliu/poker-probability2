#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_random.cuh"
#include "cuda_referencedeck.cuh"
#include "cuda_simrun.cuh"
#include "cuda_utils.cuh"

#define ADD_CARDS_64BIT 1

// Version 6:
//  - hand/cards represented by 4x uint16_t
//  - hand evaluation is the same as v3
//  - shared memory for per-thread deck storage,
//  - reference deck in shared memory as well

__device__ PokerHandType getHandType_v6(const uint8_t* hand, const CudaCard* refDeck) {
#if ADD_CARDS_64BIT
  uint64_t hv  = refDeck[hand[0]].v + refDeck[hand[1]].v + refDeck[hand[2]].v + refDeck[hand[3]].v + refDeck[hand[4]].v;
  uint16_t hv3 = static_cast<uint16_t>(hv);
  uint16_t hv2 = static_cast<uint16_t>(hv >> 16);
  uint16_t hv1 = static_cast<uint16_t>(hv >> 32);
  uint16_t hv0 = static_cast<uint16_t>(hv >> 48);
#else
  uint16_t hv0 =
      refDeck[hand[0]].v0 + refDeck[hand[1]].v0 + refDeck[hand[2]].v0 + refDeck[hand[3]].v0 + refDeck[hand[4]].v0;
  uint16_t hv1 =
      refDeck[hand[0]].v1 + refDeck[hand[1]].v1 + refDeck[hand[2]].v1 + refDeck[hand[3]].v1 + refDeck[hand[4]].v1;
  uint16_t hv2 =
      refDeck[hand[0]].v2 + refDeck[hand[1]].v2 + refDeck[hand[2]].v2 + refDeck[hand[3]].v2 + refDeck[hand[4]].v2;
  uint16_t hv3 =
      refDeck[hand[0]].v3 + refDeck[hand[1]].v3 + refDeck[hand[2]].v3 + refDeck[hand[3]].v3 + refDeck[hand[4]].v3;
#endif
  uint16_t mask = 0b0001001001001001;

  bool isFlush      = __popc(hv3 & (hv3 >> 2) & mask);
  bool isStraight   = (5 == __popc(hv2 & mask));  // Ace high straight first, so we can compute isRoyalFlush
  bool isRoyalFlush = isFlush && isStraight;
  isStraight |= (5 == __popc(hv0 & mask));
  isStraight |= (5 == __popc(hv1 & (mask >> 12)) + __popc(hv0 & (mask << 3)));
  isStraight |= (5 == __popc(hv1 & (mask >> 9)) + __popc(hv0 & (mask << 6)));
  isStraight |= (5 == __popc(hv1 & mask) + __popc(hv0 & (mask << 9)));
  isStraight |= (5 == __popc(hv2 & (mask >> 12)) + __popc(hv1 & mask) + __popc(hv0 & (mask << 12)));
  isStraight |= (5 == __popc(hv2 & (mask >> 9)) + __popc(hv1 & mask));
  isStraight |= (5 == __popc(hv2 & (mask >> 6)) + __popc(hv1 & (mask << 3)));
  isStraight |= (5 == __popc(hv2 & (mask >> 3)) + __popc(hv1 & (mask << 6)));
  isStraight |= (5 == __popc(hv2 & (mask << 12)) + __popc(hv0 & (mask >> 3)));  // Ace low straight
  bool isStraightFlush = isStraight && isFlush;
  bool isFourOfAKind   = (1 == __popc(hv0 & (mask << 2)) + __popc(hv1 & (mask << 2)) + __popc(hv2 & (mask << 2)));
  bool isThreeOfAKind  = (1 == __popc((hv0 + mask) & (mask << 2)) + __popc((hv1 + mask) & (mask << 2)) +
                                  __popc((hv2 + mask) & (mask << 2)));
  bool twoGroups       = (2 == __popc(hv0 & (mask << 1)) + __popc(hv1 & (mask << 1)) + __popc(hv2 & (mask << 1)));
  bool oneGroup        = (1 == __popc(hv0 & (mask << 1)) + __popc(hv1 & (mask << 1)) + __popc(hv2 & (mask << 1)));
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

__global__ void simKernel_v6c(curandState* states, uint64_t* globalCounts, uint32_t handsPerThread) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t shared_mem[];
  CudaCard*                 refDeck = (CudaCard*)shared_mem;
  uint8_t*                  myDeck  = &shared_mem[52 * 8 + threadIdx.x * 52];

  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < 52; ++i) refDeck[i] = getCudaCard(i);  // Initialize reference deck
  __syncthreads();

  curandState localState      = states[tid];
  uint32_t    localCounts[10] = {0};
#pragma unroll
  for (int i = 0; i < 52; ++i) myDeck[i] = i;  // Initialize local deck with cards 0..51

  for (uint64_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, myDeck);

#pragma unroll
    for (int j = 0; j < 48; ++j) {
      PokerHandType type = getHandType_v6(myDeck + j, refDeck);
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) atomicAdd(&globalCounts[i], localCounts[i]);

  states[tid] = localState;
}

__global__ void simKernel_v6x(xoro_state128* states, uint64_t* globalCounts, uint32_t handsPerThread) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t shared_mem[];
  CudaCard*                 refDeck = (CudaCard*)shared_mem;
  uint8_t*                  myDeck  = &shared_mem[52 * 8 + threadIdx.x * 52];

  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < 52; ++i) refDeck[i] = getCudaCard(i);  // Initialize reference deck
  __syncthreads();

  xoro_state128 localState      = states[tid];
  uint32_t      localCounts[10] = {0};
#pragma unroll
  for (int i = 0; i < 52; ++i) myDeck[i] = i;  // Initialize local deck with cards 0..51

  for (uint64_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, myDeck);

#pragma unroll
    for (int j = 0; j < 48; ++j) {
      PokerHandType type = getHandType_v6(myDeck + j, refDeck);
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) atomicAdd(&globalCounts[i], localCounts[i]);

  states[tid] = localState;
}

template <typename F> DeviceConfig getDeviceConfig_v6(int maxBlockSize, F func) {
  DeviceConfig config;
  int          device;
  CHECK_CUDA_ERROR(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

  // Calculate shared memory requirements for the optimized kernel
  size_t sharedMemorySize = 52 * sizeof(CudaCard)                   // Reference deck in shared memory
                            + maxBlockSize * 52 * sizeof(uint8_t);  // Local deck for each thread

  // Ensure we don't exceed device shared memory limits
  if (sharedMemorySize > prop.sharedMemPerBlock) {
    fprintf(stderr,
            "Warning: Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n",
            sharedMemorySize,
            prop.sharedMemPerBlock);
    // Adjust block size to fit shared memory
    maxBlockSize = prop.sharedMemPerBlock / ((52 * sizeof(CudaCard)) + (52 * sizeof(uint8_t)));
    maxBlockSize = (maxBlockSize / prop.warpSize) * prop.warpSize;  // Round to nearest warp size
    // Recalculate shared memory size
    sharedMemorySize = 52 * sizeof(CudaCard) + maxBlockSize * 52 * sizeof(uint8_t);
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

void simRunCUDA_v6(SimulationResult& result) {
  CUDA_DEBUG("Initializing CUDA calculation");

  // Initialize the reference deck in constant memory
  initializeReferenceDeck();

  DeviceConfig config;
  if (result.cudaRngMethod == RNGMethod::CURAND) {
    config = getDeviceConfig_v6(result.cudaBlockSize, &simKernel_v6c);
  } else {
    config = getDeviceConfig_v6(result.cudaBlockSize, &simKernel_v6x);
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

  size_t         sharedMemSize = 52 * sizeof(CudaCard) + config.blockSize * 52 * sizeof(uint8_t);
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
      simKernel_v6c<<<grid, block, sharedMemSize>>>(d_curandStates, d_counts, handsPerThread);
    } else {
      simKernel_v6x<<<grid, block, sharedMemSize>>>(d_xoroStates, d_counts, handsPerThread);
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
