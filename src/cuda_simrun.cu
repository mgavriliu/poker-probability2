#include <cuda_runtime.h>
#include "cuda_simkernel_v1.cuh"
#include "cuda_simkernel_v2.cuh"
#include "cuda_simkernel_v3.cuh"
#include "cuda_simkernel_v4.cuh"
#include "cuda_simkernel_v5.cuh"
#include "cuda_simkernel_v6.cuh"
#include "cuda_simkernel_v7.cuh"
#include "cuda_simkernel_v8.cuh"
#include "cuda_simrun.cuh"
#include "cuda_utils.cuh"
#include "utils.h"

void simRunCUDA(SimulationResult& result) {
  if (result.simVersion.size() < 6) {
    printf("Invalid simulation version: %s!\n", result.simVersion.c_str());
    return;
  }
  if (result.simVersion[6] == 'x') {
    result.cudaRngMethod = RNGMethod::XOROSHIRO;
  } else if (result.simVersion[6] == 'c') {
    result.cudaRngMethod = RNGMethod::CURAND;
  } else {
    printf("Invalid simulation version: %s!!\n", result.simVersion.c_str());
    return;
  }
  std::string version = result.simVersion.substr(0, 6);
  if (version == "gpu:v1") {
    std::cout << "Starting gpu:v1 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v1(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v2") {
    std::cout << "Starting gpu:v2 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v2(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v3") {
    std::cout << "Starting gpu:v3 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v3(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v4") {
    std::cout << "Starting gpu:v4 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v4(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v5") {
    std::cout << "Starting gpu:v5 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v5(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v6") {
    std::cout << "Starting gpu:v6 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v6(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v7") {
    std::cout << "Starting gpu:v7 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v7(result);
    std::cout << " finished!\n";
  } else if (version == "gpu:v8") {
    std::cout << "Starting gpu:v8 with " << formatNumber(result.numHands) << " hands... ";
    simRunCUDA_v8(result);
    std::cout << " finished!\n";
  } else {
    printf("Invalid simulation version: %s\n!!!", result.simVersion.c_str());
  }
}
