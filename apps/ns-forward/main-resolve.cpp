/**
 * @file main-resolve.cpp
 * @brief Solve the incompressible Navier-Stokes equations with ReSolve.
 */

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/model/ns/ForwardProblem.hpp>
#include <femx/runtime/BuildInfo.hpp>
#include <femx/runtime/Output.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>
using namespace femx;
using namespace femx::model::ns;
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

#ifndef FEMX_NS_FORWARD_APP_NAME
#define FEMX_NS_FORWARD_APP_NAME "ns-forward-resolve"
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

void setSolverOptions(ReSolveOptions& opts, const SolverParams& solver)
{
  if (solver.method == "direct")
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
  opts.max_its      = solver.max_itrs;
  opts.restart      = solver.restart;
  opts.rtol         = solver.relative_tolerance;
  opts.solve        = solver.solve;
  opts.precond      = solver.preconditioner;
  opts.gram_schmidt = solver.gram_schmidt;
  opts.sketching    = solver.sketching;
  opts.flexible     = solver.flexible;
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

int run(const Params& prm)
{
  const bool output_enabled = prm.output.enabled;

  if (output_enabled)
  {
    writeBuildInfo(prm.output.directory, makeBuildInfo());
  }

  ForwardProblem fwd(prm);

  ReSolveOptions opts;
  setSolverOptions(opts, prm.solver);

  CsrAssemblyMatrix   A(fwd.model.matrixPattern());
  ReSolveLinearSolver solver(workspaceType(prm.solver), opts);

  TimeLinearIntegrator integ(fwd.problem, A, solver);
  integ.setInitialState(fwd.x0);

  std::ofstream log_out;
  if (output_enabled)
  {
    log_out = openOutputFile(prm.output.directory, "run-info.txt");
  }

  ForwardSolveResult result;
  integ.resetTiming();
  result = solve(integ,
                 fwd,
                 prm.time,
                 prm.output,
                 &std::cout,
                 output_enabled ? &log_out : nullptr);

  if (!isFinite(result.final_state))
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
    const AppOptions opts = parseAppOptions(argc, argv, false);
    if (opts.help)
    {
      printUsage(std::cout, FEMX_NS_FORWARD_APP_NAME);
      return 0;
    }

    Params prm = loadConfig(opts.config_file);
    return run(prm);
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_NS_FORWARD_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
}
