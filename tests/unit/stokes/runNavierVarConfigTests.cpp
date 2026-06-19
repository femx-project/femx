#include <filesystem>
#include <fstream>
#include <iostream>

#include "Config.hpp"
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class NavierVarConfigTests : public TestBase
{
public:
  TestOutcome loadStraightTubeStyleConfig()
  {
    TestStatus status;
    status = true;

    const auto dir =
        std::filesystem::temp_directory_path() / "femx_navier_var_config_test";
    std::filesystem::create_directories(dir / "meshes");

    const auto config_path = dir / "Config.json";
    {
      std::ofstream config(config_path);
      config << R"({
  "simulation": {
    "mesh": {
      "file": "meshes/straighttube.msh"
    },
    "time": {
      "dt": 0.01,
      "end_time": 1.0
    },
    "fluid": {
      "rho": 1.0,
      "mu": 4.0e-7
    },
    "bcs": [
      {
        "name": "inlet",
        "physical": 4,
        "type": "dirichlet",
        "ux": 0.0,
        "uy": 0.0
      },
      {
        "name": "outlet",
        "physical": 5,
        "type": "dirichlet",
        "p": 0.0
      },
      {
        "name": "wall",
        "physical": 6,
        "type": "dirichlet",
        "ux": 0.0,
        "uy": 0.0
      }
    ],
    "output": {
      "basename": "ns-var-test"
    },
    "solver": {
      "backend": "cpu"
    }
  },
  "inverse": {
    "control": {
      "name": "inlet"
    },
    "alpha": 1.0,
    "reg": {
      "time": 1.0e-9,
      "l2": 1.0e-11
    },
    "opt": {
      "max_iterations": 12,
      "grad_abs_tolerance": 5.0e-6
    },
    "bounds": {
      "axial_min": 0.0,
      "axial_max": 0.0162,
      "normal": [1.0, 0.0, 0.0],
      "fix_non_axial": true
    },
    "initial_velocity": {
      "enabled": true,
      "lower": -0.1,
      "upper": 0.2,
      "l2": 3.0e-8
    },
    "obs": {
      "file": "obs/straighttube-obs.txt"
    }
  }
})";
    }

    const navier_var::Params prm =
        navier_var::loadConfig(config_path.string());
    const auto expected_mesh =
        (dir / "meshes/straighttube.msh").lexically_normal().string();
    if (prm.forward.mesh.file != expected_mesh)
    {
      std::cout << "mesh path was not resolved relative to config\n";
      status = false;
    }
    status           *= prm.forward.time.steps == 100;
    status           *= isEqual(prm.forward.time.dt, 0.01);
    status           *= prm.forward.bcs.size() == 3;
    status           *= prm.forward.bcs[0].name == "inlet";
    status           *= prm.forward.bcs[0].physical == 4;
    status           *= prm.forward.bcs[1].physical == 5;
    status           *= prm.forward.bcs[2].physical == 6;
    status           *= prm.inverse.control.name == "inlet";
    status           *= navier_var::pressureGauge(prm).physical == 5;
    const auto fixed  = navier_var::fixedVelocityBcs(prm);
    status           *= fixed.size() == 1;
    status           *= fixed[0].physical == 6;
    status           *= prm.forward.bcs[0].ux.has_value();
    status           *= prm.forward.bcs[0].uy.has_value();
    status           *= isEqual(*prm.forward.bcs[0].ux, 0.0);
    status           *= isEqual(*prm.forward.bcs[0].uy, 0.0);
    status           *= prm.inverse.opt.max_iterations == 12;
    status           *= prm.inverse.obs.file
                        == (dir / "obs/straighttube-obs.txt")
                               .lexically_normal()
                               .string();
    status           *= !prm.inverse.obs.grid.has_value();
    status           *= prm.inverse.obs.components.empty();
    status           *= prm.forward.output.basename == "ns-var-test";
    status           *= prm.forward.solver.backend == "cpu";
    status           *= prm.inverse.bounds.axial_max.has_value();
    status           *= isEqual(*prm.inverse.bounds.axial_max, 0.0162);
    status           *= isEqual(prm.inverse.bounds.normal[0], 1.0);
    status           *= prm.inverse.initial_velocity.enabled;
    status           *= prm.inverse.initial_velocity.lower.has_value();
    status           *= prm.inverse.initial_velocity.upper.has_value();
    status           *= isEqual(*prm.inverse.initial_velocity.lower, -0.1);
    status           *= isEqual(*prm.inverse.initial_velocity.upper, 0.2);
    status           *= isEqual(prm.inverse.initial_velocity.l2, 3.0e-8);

    const FluidParams fluid  = navier_var::fluidParams(prm);
    status                  *= isEqual(fluid.rho, 1.0);
    status                  *= isEqual(fluid.mu, 4.0e-7);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running ns-var config tests:\n";

  femx::tests::NavierVarConfigTests test;

  femx::tests::TestingResults result;
  result += test.loadStraightTubeStyleConfig();

  return result.summary();
}
