#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_random.cuh"
#include "cuda_referencedeck.cuh"
#include "cuda_simrun.cuh"
#include "cuda_utils.cuh"

#define DEBUG_MEMORY 0
#define DEBUG_MEMORY_SIZE 100
#if DEBUG_MEMORY
#define DEBUG_MEMORY_STORE(idx, value) \
  if (debug) debugMemory[idx++] = value;
#define DEBUG_MEMORY_STORE_ARRAY(idx, count, value) \
  if (debug)                                        \
    for (uint8_t i = 0; i < count; ++i) debugMemory[idx++] = value[i];
#define DEBUG_MEMORY_STORE_ARRAY_INDIRECT(idx, count, value, indirect, member) \
  if (debug)                                                                   \
    for (uint8_t i = 0; i < count; ++i) debugMemory[idx++] = value[indirect[i]].member;

void printDebugMemory(uint64_t* debugMemory);  // See the end of the file for the implementation
#else
#define DEBUG_MEMORY_STORE(idx, value)
#define DEBUG_MEMORY_STORE_ARRAY(idx, count, value)
#define DEBUG_MEMORY_STORE_ARRAY_INDIRECT(idx, count, value, indirect, member)
#endif

// Version 7:
//  - hand/cards represented by a uint64_t
//  - hand evaluation is adapted from v3
//  - shared memory for per-thread deck storage,
//  - reference deck in shared memory as well

#if DEBUG_MEMORY
__device__ PokerHandType getHandType_v7(const uint8_t* hand, const CudaCard* refDeck, uint64_t* debugMemory) {
#else
__device__ PokerHandType getHandType_v7(const uint8_t* hand, const CudaCard* refDeck) {
#endif
  uint64_t hv = refDeck[hand[0]].v + refDeck[hand[1]].v + refDeck[hand[2]].v + refDeck[hand[3]].v + refDeck[hand[4]].v;
#if DEBUG_MEMORY
  // This saves the very first random hand, but more complicated selection crieteria can be used
  bool debug  = threadIdx.x == 0 && blockIdx.x == 0;
  int  dbgIdx = 0;
#endif
  // 0-4: cardIdx[0-4]
  DEBUG_MEMORY_STORE_ARRAY(dbgIdx, 5, hand);
  // 5-9: card[0-4]
  DEBUG_MEMORY_STORE_ARRAY_INDIRECT(dbgIdx, 5, refDeck, hand, v);
  // 10: hand
  DEBUG_MEMORY_STORE(dbgIdx, hv);
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask1 = static_cast<uint64_t>(0b0000000000000001001001001000000000000000000000000000000000000000ULL);
  // 11: mask1
  DEBUG_MEMORY_STORE(dbgIdx, mask1);
  bool isFlush = (0 < __popcll(hv & (hv >> 2) & mask1));
  // 12: hv & (hv >> 2) & mask1
  DEBUG_MEMORY_STORE(dbgIdx, hv & (hv >> 2) & mask1);
  // 13: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (hv >> 2) & mask1));
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask2 = static_cast<uint64_t>(0b0000000000000000000000000000000000000000000000000001001001001001ULL);
  // 14: mask2
  DEBUG_MEMORY_STORE(dbgIdx, mask2);
  bool isStraight = (5 == __popcll(hv & (mask2 << 24)));
  // 15: hv & (mask2 << 24)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 24));
  // 16: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 24)));
  bool isRoyalFlush = isFlush && isStraight;
  isStraight |= (5 == __popcll(hv & (mask2 << 21)));
  // 17: hv & (mask2 << 21)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 21));
  // 18: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 21)));
  isStraight |= (5 == __popcll(hv & (mask2 << 18)));
  // 19: hv & (mask2 << 18)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 18));
  // 20: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 18)));
  isStraight |= (5 == __popcll(hv & (mask2 << 15)));
  // 21: hv & (mask2 << 15)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 15));
  // 22: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 15)));
  isStraight |= (5 == __popcll(hv & (mask2 << 12)));
  // 23: hv & (mask2 << 12)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 12));
  // 24: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 12)));
  isStraight |= (5 == __popcll(hv & (mask2 << 9)));
  // 25: hv & (mask2 << 9)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 9));
  // 26: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 9)));
  isStraight |= (5 == __popcll(hv & (mask2 << 6)));
  // 27: hv & (mask2 << 6)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 6));
  // 28: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 6)));
  isStraight |= (5 == __popcll(hv & (mask2 << 3)));
  // 29: hv & (mask2 << 3)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask2 << 3));
  // 30: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask2 << 3)));
  isStraight |= (5 == __popcll(hv & mask2));
  // 31: hv & mask2
  DEBUG_MEMORY_STORE(dbgIdx, hv & mask2);
  // 32: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & mask2));
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask3 = static_cast<uint64_t>(0b0000000000000000000000000001000000000000000000000000001001001001ULL);
  // 33: mask3
  DEBUG_MEMORY_STORE(dbgIdx, mask3);
  isStraight |= (5 == __popcll(hv & mask3));  // Ace low straight
  // 34: hv & mask3
  DEBUG_MEMORY_STORE(dbgIdx, hv & mask3);
  // 35: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & mask3));
  bool isStraightFlush = isStraight && isFlush;
  //                                                   |s3|s2|s1|s0|rA|rK|rQ|rJ|rT|r9|r8|r7|r6|r5|r4|r3|r2|
  uint64_t mask4 = static_cast<uint64_t>(0b0000000000000000000000000001001001001001001001001001001001001001ULL);
  // 36: mask4
  DEBUG_MEMORY_STORE(dbgIdx, mask4);
  bool isFourOfAKind = (1 == __popcll(hv & (mask4 << 2)));
  // 37: hv & (mask4 << 2)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask4 << 2));
  // 38: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask4 << 2)));
  bool isThreeOfAKind = (1 == __popcll((hv + mask4) & (mask4 << 2)));
  // 39: (hv + mask4) & (mask4 << 2)
  DEBUG_MEMORY_STORE(dbgIdx, (hv + mask4) & (mask4 << 2));
  // 40: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll((hv + mask4) & (mask4 << 2)));
  bool twoGroups = (2 == __popcll(hv & (mask4 << 1)));
  // 41: hv & (mask4 << 1)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask4 << 1));
  // 42: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask4 << 1)));
  bool oneGroup = (1 == __popcll(hv & (mask4 << 1)));
  // 43: hv & (mask4 << 1)
  DEBUG_MEMORY_STORE(dbgIdx, hv & (mask4 << 1));
  // 44: popc
  DEBUG_MEMORY_STORE(dbgIdx, __popcll(hv & (mask4 << 1)));
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

__global__ void simKernel_v7c(curandState* states,
                              uint64_t*    globalCounts,
                              uint32_t     handsPerThread
#if DEBUG_MEMORY
                              ,
                              uint64_t* debugMemory
#endif
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t shared_mem[];
  CudaCard*                 refDeck = (CudaCard*)shared_mem;
  uint8_t*                  myDeck  = &shared_mem[52 * 8 + threadIdx.x * 52];

  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < 52; ++i) refDeck[i] = getCudaCard64(i);  // Initialize reference deck
  __syncthreads();

  curandState localState      = states[tid];
  uint32_t    localCounts[10] = {0};
#pragma unroll
  for (int i = 0; i < 52; ++i) myDeck[i] = i;  // Initialize local deck with cards 0..51

  for (uint64_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, myDeck);

#pragma unroll
    for (int j = 0; j < 48; ++j) {
#if DEBUG_MEMORY
      PokerHandType type = getHandType_v7(myDeck + j, refDeck, debugMemory);
#else
      PokerHandType type = getHandType_v7(myDeck + j, refDeck);
#endif
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) atomicAdd(&globalCounts[i], localCounts[i]);

  states[tid] = localState;
}

__global__ void simKernel_v7x(xoro_state128* states,
                              uint64_t*      globalCounts,
                              uint32_t       handsPerThread
#if DEBUG_MEMORY
                              ,
                              uint64_t* debugMemory
#endif
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ uint8_t shared_mem[];
  CudaCard*                 refDeck = (CudaCard*)shared_mem;
  uint8_t*                  myDeck  = &shared_mem[52 * 8 + threadIdx.x * 52];

  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < 52; ++i) refDeck[i] = getCudaCard64(i);  // Initialize reference deck
  __syncthreads();

  xoro_state128 localState      = states[tid];
  uint32_t      localCounts[10] = {0};
#pragma unroll
  for (int i = 0; i < 52; ++i) myDeck[i] = i;  // Initialize local deck with cards 0..51

  for (uint64_t i = 0; i < handsPerThread; i += 48) {
    shuffleDeck(&localState, myDeck);

#pragma unroll
    for (int j = 0; j < 48; ++j) {
#if DEBUG_MEMORY
      PokerHandType type = getHandType_v7(myDeck + j, refDeck, debugMemory);
#else
      PokerHandType type = getHandType_v7(myDeck + j, refDeck);
#endif
      localCounts[static_cast<int>(type)]++;
    }
  }

// Write results to global memory
#pragma unroll
  for (int i = 0; i < 10; ++i) atomicAdd(&globalCounts[i], localCounts[i]);

  states[tid] = localState;
}

template <typename F> DeviceConfig getDeviceConfig_v7(int maxBlockSize, F func) {
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

void simRunCUDA_v7(SimulationResult& result) {
  CUDA_DEBUG("Initializing CUDA calculation");

  // Initialize the reference deck in constant memory
  initializeReferenceDeck64();

  DeviceConfig config;
  if (result.cudaRngMethod == RNGMethod::CURAND) {
    config = getDeviceConfig_v7(result.cudaBlockSize, &simKernel_v7c);
  } else {
    config = getDeviceConfig_v7(result.cudaBlockSize, &simKernel_v7x);
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
#if DEBUG_MEMORY
  uint64_t* d_debugMemory = nullptr;
  uint64_t* h_debugMemory = new uint64_t[100]();
#endif

  try {
    CUDA_DEBUG("Allocating device memory");
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      CHECK_CUDA_ERROR(cudaMalloc(&d_curandStates, numThreads * sizeof(curandState)));
    } else {
      CHECK_CUDA_ERROR(cudaMalloc(&d_xoroStates, numThreads * sizeof(xoro_state128)));
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_counts, 10 * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_counts, 0, 10 * sizeof(uint64_t)));
#if DEBUG_MEMORY
    CHECK_CUDA_ERROR(cudaMalloc(&d_debugMemory, 100 * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_debugMemory, 0, 100 * sizeof(uint64_t)));
#endif
    CUDA_DEBUG("Initializing RNG states");
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      initCurandRNG<<<grid, block>>>(d_curandStates, time(nullptr));
    } else {
      initXoroRNG<<<grid, block>>>(d_xoroStates, time(nullptr));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_DEBUG("Starting simulation");
#if DEBUG_MEMORY
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      simKernel_v7c<<<grid, block, sharedMemSize>>>(d_curandStates, d_counts, handsPerThread, d_debugMemory);
    } else {
      simKernel_v7x<<<grid, block, sharedMemSize>>>(d_xoroStates, d_counts, handsPerThread, d_debugMemory);
    }
#else
    if (result.cudaRngMethod == RNGMethod::CURAND) {
      simKernel_v7c<<<grid, block, sharedMemSize>>>(d_curandStates, d_counts, handsPerThread);
    } else {
      simKernel_v7x<<<grid, block, sharedMemSize>>>(d_xoroStates, d_counts, handsPerThread);
    }
#endif
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Add verification check
    uint64_t total = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(h_counts, d_counts, 10 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
#if DEBUG_MEMORY
    CHECK_CUDA_ERROR(cudaMemcpy(h_debugMemory, d_debugMemory, 100 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printDebugMemory(h_debugMemory);
#endif;

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
#if DEBUG_MEMORY
    if (d_debugMemory) cudaFree(d_debugMemory);
#endif
    delete[] h_counts;
    throw;
  }

  // Cleanup
  if (d_curandStates) cudaFree(d_curandStates);
  if (d_xoroStates) cudaFree(d_xoroStates);
  if (d_counts) cudaFree(d_counts);
#if DEBUG_MEMORY
  if (d_debugMemory) cudaFree(d_debugMemory);
#endif
  delete[] h_counts;

  CUDA_DEBUG("Calculation complete");
}

__host__ void printDebugMemory(uint64_t* debugMemory) {
  printf("Debug memory:\n");
  printf("cardIdx[0]: %2llu\n", debugMemory[0]);
  printf("cardIdx[1]: %2llu\n", debugMemory[1]);
  printf("cardIdx[2]: %2llu\n", debugMemory[2]);
  printf("cardIdx[3]: %2llu\n", debugMemory[3]);
  printf("cardIdx[4]: %2llu\n", debugMemory[4]);
  printf("card[0]:  %s\n", formatBinary64(debugMemory[5]).c_str());
  printf("card[1]:  %s\n", formatBinary64(debugMemory[6]).c_str());
  printf("card[2]:  %s\n", formatBinary64(debugMemory[7]).c_str());
  printf("card[3]:  %s\n", formatBinary64(debugMemory[8]).c_str());
  printf("card[4]:  %s\n", formatBinary64(debugMemory[9]).c_str());
  printf("PokerHand:     %s\n", formatBinary64(debugMemory[10]).c_str());
  printf("Mask1:    %s\n", formatBinary64(debugMemory[11]).c_str());
  printf("Flush:    %s\n", formatBinary64(debugMemory[12]).c_str());
  printf("Popc:     %d\n", debugMemory[13]);
  printf("Mask2:    %s\n", formatBinary64(debugMemory[14]).c_str());
  printf("Broadway: %s\n", formatBinary64(debugMemory[15]).c_str());
  printf("Popc:     %d\n", debugMemory[16]);
  printf("Straight: %s\n", formatBinary64(debugMemory[17]).c_str());
  printf("Popc:     %d\n", debugMemory[18]);
  printf("Straight: %s\n", formatBinary64(debugMemory[19]).c_str());
  printf("Popc:     %d\n", debugMemory[20]);
  printf("Straight: %s\n", formatBinary64(debugMemory[21]).c_str());
  printf("Popc:     %d\n", debugMemory[22]);
  printf("Straight: %s\n", formatBinary64(debugMemory[23]).c_str());
  printf("Popc:     %d\n", debugMemory[24]);
  printf("Straight: %s\n", formatBinary64(debugMemory[25]).c_str());
  printf("Popc:     %d\n", debugMemory[26]);
  printf("Straight: %s\n", formatBinary64(debugMemory[27]).c_str());
  printf("Popc:     %d\n", debugMemory[28]);
  printf("Straight: %s\n", formatBinary64(debugMemory[29]).c_str());
  printf("Popc:     %d\n", debugMemory[30]);
  printf("Straight: %s\n", formatBinary64(debugMemory[31]).c_str());
  printf("Popc:     %d\n", debugMemory[32]);
  printf("Mask3:    %s\n", formatBinary64(debugMemory[33]).c_str());
  printf("Straight: %s\n", formatBinary64(debugMemory[34]).c_str());
  printf("Popc:     %d\n", debugMemory[35]);
  printf("Mask4:    %s\n", formatBinary64(debugMemory[36]).c_str());
  printf("FourKind: %s\n", formatBinary64(debugMemory[37]).c_str());
  printf("Popc:     %d\n", debugMemory[38]);
  printf("ThreeKind:%s\n", formatBinary64(debugMemory[39]).c_str());
  printf("Popc:     %d\n", debugMemory[40]);
  printf("TwoGroups:%s\n", formatBinary64(debugMemory[41]).c_str());
  printf("Popc:     %d\n", debugMemory[42]);
  printf("OneGroup: %s\n", formatBinary64(debugMemory[43]).c_str());
  printf("Popc:     %d\n", debugMemory[44]);
}
