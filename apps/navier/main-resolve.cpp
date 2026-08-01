/**
 * @file main-resolve.cpp
 * @brief Solve the incompressible Navier-Stokes equations with ReSolve.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "CommandLine.hpp"
#include "Config.hpp"
#include "NavierProblem.hpp"
#include "Solve.hpp"
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/model/navier/NavierResidual.hpp>
#include <femx/runtime/BuildInfo.hpp>
#include <femx/runtime/Output.hpp>
#include <femx/state/TimeIntegrator.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#else
#include <femx/linalg/host/HostLinearSystem.hpp>
#endif

using namespace femx;
using namespace femx::apps::navier;
using namespace femx::state;
using namespace femx::linalg;
using namespace femx::runtime;

#ifndef FEMX_GIT_COMMIT
#define FEMX_GIT_COMMIT "unknown"
#endif

#ifndef FEMX_CMAKE_BUILD_TYPE
#define FEMX_CMAKE_BUILD_TYPE ""
#endif

#ifndef FEMX_CMAKE_CXX_COMPILER
#define FEMX_CMAKE_CXX_COMPILER "unknown"
#endif

#ifndef FEMX_CMAKE_CUDA_ARCHITECTURES
#define FEMX_CMAKE_CUDA_ARCHITECTURES ""
#endif

#ifndef FEMX_ENABLE_HDF5_OPTION
#define FEMX_ENABLE_HDF5_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_OPENMP_OPTION
#define FEMX_ENABLE_OPENMP_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_RESOLVE_OPTION
#define FEMX_ENABLE_RESOLVE_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_ENZYME_OPTION
#define FEMX_ENABLE_ENZYME_OPTION "unknown"
#endif

#ifndef FEMX_NAVIER_APP_NAME
#define FEMX_NAVIER_APP_NAME "navier-resolve"
#endif

namespace
{

BuildInfo makeBuildInfo()
{
  return BuildInfo{
      {{"femx commit", FEMX_GIT_COMMIT},
       {"cmake build type", FEMX_CMAKE_BUILD_TYPE},
       {"cmake cxx compiler", FEMX_CMAKE_CXX_COMPILER},
       {"cmake cuda architectures", FEMX_CMAKE_CUDA_ARCHITECTURES},
       {"FEMX_ENABLE_HDF5", FEMX_ENABLE_HDF5_OPTION},
       {"FEMX_ENABLE_OPENMP", FEMX_ENABLE_OPENMP_OPTION},
       {"FEMX_ENABLE_RESOLVE", FEMX_ENABLE_RESOLVE_OPTION},
       {"FEMX_ENABLE_ENZYME", FEMX_ENABLE_ENZYME_OPTION}}};
}

void setSolverOptions(ReSolveOptions& opts, const SolverConfig& prm)
{
  if (prm.method == "direct")
  {
    opts          = ReSolveOptions{};
    opts.factor   = "klu";
    opts.refactor = "none";
    opts.solve    = "klu";
    opts.precond  = "none";
    opts.ir       = "none";
    return;
  }

  opts.factor       = "none";
  opts.refactor     = "none";
  opts.ir           = "none";
  opts.max_its      = prm.max_itrs;
  opts.restart      = prm.restart;
  opts.rtol         = prm.relative_tolerance;
  opts.solve        = prm.solve;
  opts.precond      = prm.preconditioner;
  opts.gram_schmidt = prm.gram_schmidt;
  opts.sketching    = prm.sketching;
  opts.flexible     = prm.flexible;
}

int run(const Config& prm)
{
  const bool output_enabled = prm.output.enabled;

  if (output_enabled)
  {
    writeBuildInfo(prm.output.directory, makeBuildInfo());
  }

  NavierProblem problem(prm);

  ReSolveOptions opts;
  setSolverOptions(opts, prm.solver);

  std::ofstream log_out;
  if (output_enabled)
  {
    log_out = openOutputFile(prm.output.directory, "run-info.txt");
  }

  SolveResult result;

#if defined(FEMX_RESOLVE_USE_CUDA)

  auto             solver = std::make_unique<ReSolveLinearSolver>(opts);
  CudaLinearSystem system(std::move(solver));

  auto& ctx = static_cast<linalg::CudaContext&>(system.context());

  model::navier::CudaNavierResidual       navier(problem.model(), ctx);
  assembly::DeviceConstrainedTimeResidual res(navier, problem.controlMap(), {}, ctx);

  DeviceTimeIntegrator integ(res, system);

  DeviceVector<Real> init_state;
  ctx.vectorHandler().copy(problem.initialState(), init_state);
  ctx.sync();

  integ.setInitialState(init_state);
  result = solve(integ,
                 problem,
                 prm.time,
                 prm.output,
                 &std::cout,
                 output_enabled ? &log_out : nullptr);

#else

  auto             solver = std::make_unique<ReSolveLinearSolver>(opts);
  HostLinearSystem system(std::move(solver));

  model::navier::HostNavierResidual     navier(problem.model());
  assembly::HostConstrainedTimeResidual res(navier, problem.controlMap());

  HostTimeIntegrator integ(res, system);

  integ.setInitialState(problem.initialState());
  result = solve(integ,
                 problem,
                 prm.time,
                 prm.output,
                 &std::cout,
                 output_enabled ? &log_out : nullptr);

#endif

  if (!hasFiniteValues(result.final_state))
  {
    throw std::runtime_error("Linear solve produced non-finite values in x");
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    const CommandLineOptions opts = parseCommandLine(argc, argv, false);
    if (opts.help)
    {
      printUsage(std::cout, FEMX_NAVIER_APP_NAME);
      return 0;
    }

    const Config prm = loadConfig(opts.config_file);
    return run(prm);
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_NAVIER_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
}
