#pragma once

#include "cuda_utils.cuh"
#include "random.h"
#include "simulation.h"

struct DeviceConfig {
  int blockSize;
  int numBlocks;
  int maxThreadsPerSM;
  int numSMs;
};

void simRunCUDA(SimulationResult& result);
void initializeReferenceDeck();