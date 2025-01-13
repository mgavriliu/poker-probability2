#include "simulation.h"
#include <memory>
#include "simulation_v1.h"
#include "simulation_v2.h"

std::unique_ptr<Simulation> Simulation::getSimulation(SimulationResult& result) {
  if (result.simVersion == "cpu:v1") {
    return std::make_unique<SimulationV1>(result);
  } else if (result.simVersion == "cpu:v2") {
    return std::make_unique<SimulationV2>(result);
  } else {
    std::cerr << "Invalid simulation type: " << result.simVersion << std::endl;
    exit(EXIT_FAILURE);
  }
}
