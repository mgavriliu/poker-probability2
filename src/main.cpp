#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "cuda_simrun.cuh"
#include "random.h"
#include "simulation.h"
#include "utils.h"

// Common functionality implemented in base class
void printResults(SimulationResult result) {
  std::cout << "\n";
  std::cout << "Implementation: " << result.simVersion << "\n";
  std::cout << " -- settings: [mult: " << result.handsMultiplier << "x";
  if (result.simVersion.substr(0, 4) == "gpu:") {
    std::cout << ", RNG: " << (result.cudaRngMethod == RNGMethod::CURAND ? "curand" : "xoro");
    std::cout << ", block size: " << result.cudaBlockSize;
    std::cout << ", blocks: " << result.cudaNumBlocks;
  }
  std::cout << ", threads: " << result.numThreads << "]\n";
  std::cout << std::string(105, '=') << "\n";
  std::cout << std::left << std::setw(20) << "PokerHand Type" << std::right << std::setw(20) << "Count" << std::setw(15)
            << "Calculated" << std::setw(15) << "Theoretical" << std::setw(15) << "Error" << std::setw(15) << "Sigma"
            << "\n";
  std::cout << std::string(105, '-') << "\n";

  double   sumSquaredErrors = 0.0;
  uint64_t actualHands      = 0;
  for (const auto& count : result.counts) actualHands += count;

  for (int t = 0; t < static_cast<int>(PokerHandType::Count); ++t) {
    PokerHandType type        = static_cast<PokerHandType>(t);
    double        prob        = result.getProbability(type);
    double        theoretical = Poker::getTheoreticalProbability(type);
    double        error       = (prob * 100) - theoretical;

    // Calculate standard error for binomial distribution
    double p              = theoretical / 100.0;  // Convert percentage to probability
    double standardError  = std::sqrt((p * (1.0 - p)) / actualHands) * 100.0;
    double sigmaDeviation = error / standardError;
    sumSquaredErrors += error * error;  // Add squared error

    std::cout << std::left << std::setw(20) << Poker::toString(type) << std::right << std::setw(20)
              << formatNumber(result.counts[t]) << std::fixed << std::setprecision(6) << std::setw(14) << (prob * 100)
              << "%" << std::setw(14) << theoretical << "%" << std::setw(14) << error << "%" << std::setw(15)
              << sigmaDeviation << "\n";
  }

  double stdDev = std::sqrt(sumSquaredErrors / static_cast<double>(static_cast<int>(PokerHandType::Count)));

  std::cout << std::string(105, '-') << "\n";
  std::cout << std::left << std::setw(20) << "Total:" << std::right << std::setw(20) << formatNumber(result.numHands)
            << "\nTime: " << std::fixed << std::setprecision(2) << result.elapsedTime << "s"
            << "\nSpeed: " << formatNumber(static_cast<uint64_t>(actualHands / result.elapsedTime)) << " hands/s"
            << "\nRMS Error: " << std::fixed << std::setprecision(6) << stdDev << "%\n";
  std::cout << std::string(105, '=') << "\n\n";
}

void printUsage(const char* programName) {
  std::cout
      << "Usage: " << programName << " [options]\n"
      << "Options:\n"
      << "  -h, --help              Show this help message\n"
      << "  -n NUMBER               Number of hands to simulate (default: 100000000)\n"
      << "  -b, --blocksize         Global block size for GPU implementations (default: 32)\n"
      << "  -i, --implementation    Comma separated list of versions to run (default: cpu:v1)\n"
      << "        The format is 'TTT:vvv[:sssx:gblk]' where TTT is the type (cpu or gpu),\n"
      << "        vvv is the version, sssx is a scaling factor, and gblk is the block size (GPU only.)\n"
      << "        The type and version are required, while the scaling and block sizes are optional.\n"
      << "        The scaling factor will multiply the number of hands by the specified factor.\n"
      << "        The block size is only used for GPU versions, and it overrides the global block size\n"
      << "        specified with the -g or --blocksize command line options.\n"
      << "        Available versions:\n"
      << "         all     - runs all versions with the same number of hands and xoroshiro random number gen.\n"
      << "         cpu:v1  - hand:8b,    eval:logic, rng:xoro\n"
      << "         cpu:v2  - hand:64b,   eval:mask,  rng:xoro\n"
      << "         gpu:v1c - hand:8b,    eval:logic, deck:local,  rng:curand\n"
      << "         gpu:v1x - hand:8b,    eval:logic, deck:local,  rng:xoro\n"
      << "         gpu:v2c - hand:8b,    eval:logic, deck:shared, rng:curand\n"
      << "         gpu:v2x - hand:8b,    eval:logic, deck:shared, rng:xoro\n"
      << "         gpu:v3c - hand:4x16b, eval:mask,  deck:shared, conv:algo,   rng:curand\n"
      << "         gpu:v3x - hand:4x16b, eval:mask,  deck:shared, conv:algo,   rng:xoro\n"
      << "         gpu:v4c - hand:4x16b, eval:mask2, deck:shared, conv:lookup, rng:curand, ref:const\n"
      << "         gpu:v4x - hand:4x16b, eval:mask2, deck:shared, conv:lookup, rng:xoro,   ref:const\n"
      << "         gpu:v5c - hand:4x16b, eval:mask,  deck:shared, conv:lookup, rng:curand, ref:const\n"
      << "         gpu:v5x - hand:4x16b, eval:mask,  deck:shared, conv:lookup, rng:xoro,   ref:const\n"
      << "         gpu:v6c - hand:4x16b, eval:mask,  deck:shared, conv:lookup, rng:curand, ref:shared\n"
      << "         gpu:v6x - hand:4x16b, eval:mask,  deck:shared, conv:lookup, rng:xoro,   ref:shared\n"
      << "         gpu:v7c - hand:64b,   eval:mask,  deck:shared, conv:lookup, rng:curand, ref:shared\n"
      << "         gpu:v7x - hand:64b,   eval:mask,  deck:shared, conv:lookup, rng:xoro,   ref:shared\n"
      << "         gpu:v8c - hand:64b,   eval:mask,  deck:shared, conv:lookup, rng:curand, ref:shared, counts:64b-shared\n"
      << "         gpu:v8x - hand:64b,   eval:mask,  deck:shared, conv:lookup, rng:xoro,   ref:shared, counts:64b-shared\n"
      << std::endl;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream        ss(str);
  std::string              token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

SimulationResult runSimulation(const std::string& version,
                               uint64_t           hands,
                               uint64_t           scale,
                               int                blockSize,
                               RNGMethod          rngMethod) {
  SimulationResult result;
  result.simVersion      = version;
  result.numHands        = hands * scale;
  result.handsMultiplier = scale;
  result.cudaBlockSize   = blockSize;
  result.cudaRngMethod   = rngMethod;

  if (version.substr(0, 4) == "cpu:") {
    auto sim = Simulation::getSimulation(result);
    sim->run();
    result             = sim->getResults();
    result.elapsedTime = sim->getElapsedTime();
  } else if (version.substr(0, 4) == "gpu:") {
    auto start = std::chrono::high_resolution_clock::now();
    simRunCUDA(result);
    auto end           = std::chrono::high_resolution_clock::now();
    result.elapsedTime = std::chrono::duration<double>(end - start).count();
  }
  return result;
}

void printPerformanceComparison(const std::vector<SimulationResult>& results) {
  // Sort results by performance (hands/s)
  std::vector<SimulationResult> sorted = results;
  std::sort(sorted.begin(), sorted.end(), [](const SimulationResult& a, const SimulationResult& b) {
    return a.getHandsPerSecond() > b.getHandsPerSecond();
  });

  double slowestSpeed = sorted.back().getHandsPerSecond();

  std::cout << "\nPerformance Comparison:\n";
  std::cout << std::string(85, '=') << "\n";
  std::cout << std::left << std::setw(15) << "Implementation" << std::right << std::setw(15) << "Time(s)"
            << std::setw(20) << "Hands/s" << std::setw(15) << "Speedup" << std::setw(20) << "Relative Perf.\n";
  std::cout << std::string(85, '-') << "\n";

  for (const auto& result : sorted) {
    double speedup = result.getHandsPerSecond() / slowestSpeed;
    std::cout << std::left << std::setw(15) << result.simVersion << std::right << std::fixed << std::setprecision(2)
              << std::setw(15) << result.elapsedTime << std::setw(20)
              << formatNumber(static_cast<uint64_t>(result.getHandsPerSecond())) << std::setw(15) << speedup
              << std::setw(20) << (speedup * 100) << "%\n";
  }
  std::cout << std::string(85, '=') << "\n";
}

int main(int argc, char* argv[]) {
  std::locale::global(std::locale(""));
  std::cout.imbue(std::locale(""));

  uint64_t    totalHands      = 100'000'000;
  RNGMethod   rngMethod       = RNGMethod::CURAND;
  bool        useFisherYates  = true;
  int         cudaBlockSize   = 64;
  std::string implementations = "cpu:v1";

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "-n" && i + 1 < argc) {
      totalHands = std::stoull(argv[++i]);
    } else if (arg == "-b" && i + 1 < argc) {
      cudaBlockSize = std::max(32, std::stoi(argv[++i]));
    } else if ((arg == "-i" || arg == "--implementation") && i + 1 < argc) {
      implementations = argv[++i];
    }
  }

  std::vector<std::string> versions;
  if (implementations == "all") {
    versions = {
        "cpu:v1", "cpu:v2", "gpu:v1x", "gpu:v2x", "gpu:v3x", "gpu:v4x", "gpu:v5x", "gpu:v6x", "gpu:v7x", "gpu:v8x"};
  } else {
    versions = splitString(implementations, ',');
  }

  std::vector<SimulationResult> results;
  for (const auto& version : versions) {
    auto     parts = splitString(version, ':');
    uint64_t scale = 1;
    if (parts.size() >= 3) {
      if (parts[2].back() == 'x') {
        scale = std::stoull(parts[2].substr(0, parts[2].size() - 1));
      }
    }
    int localCudaBlockSize = cudaBlockSize;
    if (parts.size() >= 4) {
      localCudaBlockSize = std::stoi(parts[3]);
    }
    results.push_back(runSimulation(parts[0] + ":" + parts[1], totalHands, scale, localCudaBlockSize, rngMethod));
    // Print individual results
    printResults(results.back());
  }

  // Print performance comparison if multiple versions were run
  if (results.size() > 1) {
    printPerformanceComparison(results);
  }

  return 0;
}