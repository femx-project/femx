/**
 * @file main-resolve.cpp
 * @brief Solve the incompressible Navier-Stokes equations with ReSolve.
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Assembly.hpp"
#include "BoundaryConditions.hpp"
#include "Config.hpp"
#include "RunSupport.hpp"
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/mesh/GmshReader.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <femx/system/resolve/ReSolveLinearSolver.hpp>

using namespace femx;
using namespace femx::assembly;
using namespace femx::system;

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
       {"FEMX_ENABLE_RESOLVE", FEMX_ENABLE_RESOLVE_OPTION}}};
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

int run(const Params& params, bool enable_output)
{
  if (enable_output)
  {
    writeBuildInfo(params.output, makeBuildInfo());
  }

  Mesh mesh = GmshReader::read(params.mesh_file);
  auto elem = makeElem(mesh, "navier-gls");

  FESpace u_space(&mesh, elem.get(), mesh.dim());
  FESpace p_space(&mesh, elem.get());

  MixedFESpace space;
  space.addField(u_space);
  space.addField(p_space);
  space.setup();

  auto               pattern = SparsityPatternBuilder::build(space);
  SparseSystemMatrix A(pattern);
  Vector             b(space.numDofs());
  Vector             x(space.numDofs());
  Vector             xp(space.numDofs());
  x.setZero();
  xp.setZero();

  ReSolveOptions opts;
  setSolverOptions(opts);

  auto                work = workspaceType(params.solver);
  ReSolveLinearSolver solver(work, opts);

  std::vector<Snapshot> snapshots;
  std::ofstream         run_log;
  if (enable_output)
  {
    run_log = openRunLog(params.output);
  }

  std::cout << "navier-gls: ranks = 1"
            << ", dofs = " << space.numDofs()
            << ", cells = " << space.mesh().numElems() << '\n';

  for (Index step = 1; step <= params.time.steps; ++step)
  {
    const Real time       = step * params.time.dt;
    const auto step_start = Clock::now();

    AssemblyStats stats;
    const double  asm_time = timeBlock(
        [&]
        {
          assembleSystem(space,
                         x,
                         xp,
                         step == 1,
                         params.fluid,
                         params.time.dt,
                         A,
                         b,
                         stats);
        });

    if (!std::isfinite(stats.max_cfl))
    {
      throw std::runtime_error("Stopping as CFL became invalid");
    }

    Vector       x_old   = x;
    const auto   bc      = makeBoundaryCondition(space, params.bcs, time);
    const double bc_time = timeBlock(
        [&]
        {
          bc.apply(A.matrix(), b);
        });

    const double solve_time = timeBlock(
        [&]
        {
          solver.setOperator(A.matrix());
          solver.solve(b, x);
        });

    if (!isFinite(x))
    {
      throw std::runtime_error("Linear solve produced non-finite values in x");
    }
    xp = x_old;

    double out_time = 0.0;
    if (enable_output && shouldWriteOutput(step, params.time.steps, params.output))
    {
      out_time = timeBlock(
          [&]
          {
            snapshots.push_back(makeSnapshot(space, x, time));
            writeOutput(mesh, params.output, snapshots);
          });
    }

    const double       total_time = elapsedSeconds(step_start, Clock::now());
    std::ostringstream line;
    line << "step " << std::setw(7) << step << " / " << std::setw(7)
         << params.time.steps << ", t = " << std::setw(11) << time
         << ", max CFL = " << std::setw(11) << stats.max_cfl
         << ", assembly = " << std::setw(11) << asm_time << " s"
         << ", bc = " << std::setw(11) << bc_time << " s"
         << ", solve = " << std::setw(11) << solve_time << " s"
         << ", output = " << std::setw(11) << out_time << " s"
         << ", total = " << std::setw(11) << total_time << " s";
    std::cout << line.str() << '\n';

    if (enable_output)
    {
      run_log << "step " << std::setw(7) << step << ", t = "
              << std::setw(11) << time << ", max CFL = "
              << std::setw(11) << stats.max_cfl
              << ", assembly = " << asm_time
              << ", bc = " << bc_time
              << ", solve = " << solve_time
              << ", output = " << out_time
              << ", total = " << total_time << '\n';
      if (shouldWriteOutput(step, params.time.steps, params.output))
      {
        run_log.flush();
      }
    }
  }

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
      printUsage(std::cout, "navier-gls");
      return 0;
    }

    Params params = loadConfig(options.config_file);
    if (options.steps)
    {
      params.time.steps = *options.steps;
    }
    return run(params, !options.no_output);
  }
  catch (const std::exception& e)
  {
    std::cerr << "navier-gls: " << e.what() << '\n';
    return 1;
  }
}
