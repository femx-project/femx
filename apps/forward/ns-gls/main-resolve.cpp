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
#include <femx/solve/TimeLinearStateSolver.hpp>
#include <femx/solve/TimeTrajectory.hpp>

using namespace femx;
using namespace femx::linalg;
using namespace femx::solve;

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

void setSolverOptions(ReSolveOptions& options)
{
  options.factor   = "none";
  options.refactor = "none";
  options.ir       = "none";
  options.max_its  = 5000;
  options.restart  = 200;
  options.rtol     = 1.0e-8;
  options.solve    = "fgmres";
  options.precond  = "ilu0";
  options.flexible = true;
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

void writeRunSummary(std::ofstream&               run_log,
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

  ForwardProblem problem(prm);

  ReSolveOptions opts;
  setSolverOptions(opts);

  SparseMatrixOperator next_jac(problem.pattern);
  ReSolveLinearSolver  solver(workspaceType(prm.solver), opts);

  TimeLinearStateSolver state_solver(problem.eq, next_jac, solver);
  state_solver.setInitialState(problem.x0);
  state_solver.setStepMonitor(
      [](Index step, Index total)
      {
        std::cout << "\r  time step " << std::setw(7) << step << " / "
                  << std::setw(7) << total << std::flush;
        if (step >= total)
        {
          std::cout << '\n';
        }
      });

  std::cout << FEMX_NAVIER_GLS_APP_NAME << ": ranks = 1"
            << ", dofs = " << problem.space.numDofs()
            << ", cells = " << problem.space.mesh().numElems() << '\n';

  TimeTrajectory trajectory;
  state_solver.resetTiming();
  const double total_time = timeBlock(
      [&]
      {
        state_solver.solve(problem.prm0, trajectory);
      });

  if (!isFinite(trajectory[problem.steps]))
  {
    throw std::runtime_error("Linear solve produced non-finite values in x");
  }

  if (enable_output)
  {
    writeTrajectoryOutput(problem, trajectory, prm.output);
    std::ofstream run_log = openRunLog(prm.output);
    writeRunSummary(run_log, problem, state_solver, total_time);
  }

  std::cout << "  assembly = " << state_solver.assemblySeconds() << " s"
            << ", solve = " << state_solver.solveSeconds() << " s"
            << ", total = " << total_time << " s" << '\n';

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    const AppOptions options = parseAppOptions(argc, argv, false);
    if (options.help)
    {
      printUsage(std::cout, FEMX_NAVIER_GLS_APP_NAME);
      return 0;
    }

    Params prm = loadConfig(options.config_file);
    if (options.steps)
    {
      prm.time.steps = *options.steps;
    }
    return run(prm, !options.no_output);
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_NAVIER_GLS_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
}
