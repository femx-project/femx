/**
 * @file main-resolve.cpp
 * @brief Solve the incompressible Navier-Stokes equations with ReSolve.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "RunSupport.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/SparseMatrixOperator.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>

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

  opts.factor   = "none";
  opts.refactor = "none";
  opts.ir       = "none";
  opts.max_its  = 5000;
  opts.restart  = 200;
  opts.rtol     = 1.0e-8;
  opts.solve    = "fgmres";
  opts.precond  = "ilu0";
  opts.flexible = true;
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

void writeRunSummary(ofstream&                    run_log,
                     const ForwardProblem&        problem,
                     const TimeLinearStateSolver& state_solver,
                     Real                         total_seconds)
{
  run_log << "steps = " << problem.steps << '\n'
          << "dt = " << problem.dt << '\n'
          << "dofs = " << problem.space.numDofs() << '\n'
          << "cells = " << problem.space.mesh().numElems() << '\n'
          << "assembly calls = " << state_solver.assemblyCalls() << '\n'
          << "solve calls = " << state_solver.solveCalls() << '\n'
          << "assembly seconds = " << state_solver.assemblySeconds() << '\n'
          << "solve seconds = " << state_solver.solveSeconds() << '\n'
          << "total seconds = " << total_seconds << '\n';
}

int run(const Params& prm, bool enable_output)
{
  if (enable_output)
  {
    writeBuildInfo(prm.output, makeBuildInfo());
  }

  ForwardProblem forward(prm);

  ReSolveOptions opts;
  setSolverOptions(opts, prm.solver);

  SparseMatrixOperator next_jac(forward.pat);
  ReSolveLinearSolver  solver(workspaceType(prm.solver), opts);

  TimeLinearStateSolver state_solver(forward.problem, next_jac, solver);
  state_solver.setInitialState(forward.x0);
  state_solver.setStepMonitor(
      [](Index step, Index total)
      {
        cout << "\r  time step " << setw(7) << step << " / "
             << setw(7) << total << flush;
        if (step >= total)
        {
          cout << '\n';
        }
      });

  cout << FEMX_NAVIER_GLS_APP_NAME << ": ranks = 1"
       << ", dofs = " << forward.space.numDofs()
       << ", cells = " << forward.space.mesh().numElems() << '\n';

  ForwardSolveResult result;
  state_solver.resetTiming();
  const double total_time = timeBlock(
      [&]
      {
        result = solve(state_solver, forward, prm.output, enable_output);
      });

  if (!isFinite(result.final_state))
  {
    throw runtime_error("Linear solve produced non-finite values in x");
  }

  if (enable_output)
  {
    if (!result.snapshots.empty())
    {
      writeOutput(forward.mesh, prm.output, result.snapshots);
    }
    ofstream run_log = openRunLog(prm.output);
    writeRunSummary(run_log, forward, state_solver, total_time);
  }

  cout << "  assembly = " << state_solver.assemblySeconds() << " s"
       << ", solve = " << state_solver.solveSeconds() << " s"
       << ", total = " << total_time << " s" << '\n';

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
