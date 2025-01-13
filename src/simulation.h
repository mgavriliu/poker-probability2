#pragma once

#include <array>
#include <chrono>
#include <iostream>
#include "poker.h"
#include "random.h"
#include "utils.h"

struct SimulationResult {
  std::string                                                      simVersion;
  int                                                              cudaNumBlocks   = 0;
  int                                                              cudaBlockSize   = 0;
  RNGMethod                                                        cudaRngMethod   = RNGMethod::CURAND;
  int                                                              numThreads      = 1;
  uint64_t                                                         numHands        = 100000000;
  int                                                              handsMultiplier = 1;
  double                                                           elapsedTime     = 0.0;
  std::array<uint64_t, static_cast<uint8_t>(PokerHandType::Count)> counts{0};

  inline void   addHand(PokerHandType type) { counts[static_cast<uint8_t>(type)]++; }
  inline double getProbability(PokerHandType type) const {
    uint64_t total = 0;
    for (const auto& count : counts) total += count;
    return total > 0 ? static_cast<double>(counts[static_cast<uint8_t>(type)]) / total : 0.0;
  }
  double getHandsPerSecond() const { return numHands / elapsedTime; }
};

// Forward declarations
class SimulationV1;
class SimulationV2;

class Simulation {
 public:
  virtual ~Simulation() = default;

  // Pure virtual functions that must be implemented by derived classes
  virtual std::string id()  = 0;
  virtual void        run() = 0;

  static std::unique_ptr<Simulation> getSimulation(SimulationResult& result);

  SimulationResult getResults() const { return result; }
  double           getElapsedTime() const { return result.elapsedTime; }

 protected:
  // Common member variables available to derived classes
  SimulationResult result;

  // Utility functions for derived classes
  inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
};

// Include derived class definitions after base class
#include "simulation_v1.h"
#include "simulation_v2.h"