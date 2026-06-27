/**
 * @file main-resolve.cpp
 * @brief Solve the incompressible Navier-Stokes equations with ReSolve.
 */

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "RunSupport.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/SparseMatrixOperator.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>

using namespace std;
using namespace femx;
using namespace femx::state;
using namespace femx::linalg;

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

#ifndef FEMX_NAVIER_GLS_APP_NAME
#define FEMX_NAVIER_GLS_APP_NAME "ns-gls"
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
    opts = ReSolveOptions{};
    return;
  }

  opts.factor              = "none";
  opts.refactor            = "none";
  opts.ir                  = "none";
  opts.max_its             = solver.max_iterations;
  opts.restart             = solver.restart;
  opts.rtol                = solver.relative_tolerance;
  opts.solve               = solver.solve;
  opts.precond             = solver.preconditioner;
  opts.gram_schmidt        = solver.gram_schmidt;
  opts.sketching           = solver.sketching;
  opts.preconditioner_side = solver.preconditioner_side;
  opts.flexible            = solver.flexible;
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

int run(const Params& prm, bool enable_output)
{
  if (enable_output)
  {
    writeBuildInfo(prm.output, makeBuildInfo());
  }

  ForwardProblem fwd(prm);

  ReSolveOptions opts;
  setSolverOptions(opts, prm.solver);

  SparseMatrixOperator A(fwd.pettern);
  ReSolveLinearSolver  solver(workspaceType(prm.solver), opts);

  TimeLinearIntegrator integrator(fwd.problem, A, solver);
  integrator.setInitialState(fwd.x0);

  cout << FEMX_NAVIER_GLS_APP_NAME << ": ranks = 1"
       << ", dofs = " << fwd.space.numDofs()
       << ", elems = " << fwd.space.mesh().numElems() << '\n';

  ofstream log_out;
  if (enable_output)
  {
    log_out = openRunLog(prm.output);
  }

  ForwardSolveResult result;
  integrator.resetTiming();
  result = solve(integrator,
                 fwd,
                 prm.time,
                 prm.output,
                 enable_output,
                 &cout,
                 enable_output ? &log_out : nullptr);

  if (!isFinite(result.final_state))
  {
    throw runtime_error("Linear solve produced non-finite values in x");
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
      printUsage(cout, FEMX_NAVIER_GLS_APP_NAME);
      return 0;
    }

    Params prm = loadConfig(opts.config_file);
    if (opts.steps)
    {
      prm.time.steps = *opts.steps;
    }
    return run(prm, !opts.no_output);
  }
  catch (const exception& e)
  {
    cerr << FEMX_NAVIER_GLS_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
}
