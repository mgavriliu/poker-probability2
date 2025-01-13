#pragma once

#include <array>
#include <ctime>
#include <iostream>
#include "poker.h"
#include "simulation.h"
#include "threads.h"
#include "utils.h"

class SimulationV2 : public Simulation {
 public:
  SimulationV2(const SimulationResult& params) { this->result = params; }
  ~SimulationV2() = default;

  std::string id() override { return "cpu:v2"; }

  void run() {
    uint64_t totalHands = this->result.numHands;
    int      numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::cout << "\nStarted " << this->id() << " with " << formatNumber(totalHands) << " hands..." << std::endl;
#ifdef NDEBUG
    std::cout << ", threads: " << numThreads << "\n";
#else
    std::cout << " (Debug mode, single threaded.)\n";
#endif
    // Keep time for benchmarking
    auto start = std::chrono::high_resolution_clock::now();
#ifdef NDEBUG
    // Multi-threaded implementation for release builds
    ThreadPool                    pool(numThreads);
    uint64_t                      handsPerThread = totalHands / numThreads;
    uint64_t                      remainder      = totalHands % numThreads;
    std::vector<SimulationResult> localCounts(numThreads);

    for (int i = 0; i < numThreads; ++i) {
      uint64_t handsToSimulate = handsPerThread + (i < remainder ? 1 : 0);
      pool.enqueue([this, &localCounts, i, handsToSimulate]() { localCounts[i] = simKernel(handsToSimulate, i); });
    }

    // Wait for all tasks to complete
    pool.~ThreadPool();

    // Aggregate results
    for (const auto& localCount : localCounts) {
      for (size_t i = 0; i < static_cast<size_t>(PokerHandType::Count); ++i) {
        result.counts[i] += localCount.counts[i];
      }
    }
    result.numThreads = numThreads;
#else
    // Single-threaded implementation for debug builds
    SimulationResult localCount = simKernel(result.numHands);
    for (size_t i = 0; i < static_cast<size_t>(PokerHandType::Count); ++i) {
      this->result.counts[i] += localCount.counts[i];
    }
    result.numThreads = 1;
#endif
    auto end           = std::chrono::high_resolution_clock::now();
    result.elapsedTime = std::chrono::duration<double>(end - start).count();
  }

 private:
  SimulationResult simKernel(uint64_t handsToSimulate, int threadId = 0) {
    SimulationResult localCounts;
    uint8_t          deck[52] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51};
    // Seed with thread ID plus time for better distribution
    // Xoroshiro128+ state
    struct {
      uint64_t s[2];
    } xoro_state;
    // Initialize xoro_state using splitmix64
    uint64_t seed   = static_cast<uint64_t>(threadId) + static_cast<uint64_t>(time(nullptr));
    uint64_t z      = (seed + 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    xoro_state.s[0] = z ^ (z >> 31);

    z               = (seed + 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    xoro_state.s[1] = z ^ (z >> 31);

    for (uint64_t j = 0; j < handsToSimulate; j += 48) {
      // Shuffle the deck using Xoroshiro128+
      for (int i = 51; i > 0; i -= 2) {
        const uint64_t s0   = xoro_state.s[0];
        uint64_t       s1   = xoro_state.s[1];
        uint64_t       rand = s0 + s1;

        s1 ^= s0;
        xoro_state.s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        xoro_state.s[1] = rotl(s1, 37);
        // Since rand is 64 bits, and we only need an 8 bit random number
        // for each iteration of the loop. We can use all 8 random bytes
        // available in the 64 bit random number before asking for a new one.
#pragma unroll
        for (int byte = 0; byte < 8 && i - byte >= 0; ++byte) {
          // Extract each byte and use it to generate an index
          uint8_t randByte = (rand >> (byte * 8)) & 0xFF;
          // Map the byte to a valid index range (0 to i)
          int j = randByte % (i - byte + 1);
          std::swap(deck[i - byte], deck[j]);
        }
      }
      for (int k = 0; k < 48; ++k) {
        PokerHand64 hand = PokerHand64(deck + k, 5);
        localCounts.counts[static_cast<uint64_t>(hand.ComputeHandRank())]++;
      }
    }
    return localCounts;
  }
};
