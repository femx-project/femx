#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly.hpp"
#include "BoundaryConditions.hpp"
#include "Config.hpp"
#include "RunSupport.hpp"
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/mesh/GmshReader.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/petsc/KspLinearSolver.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>
#include <femx/system/petsc/PETScSystemVector.hpp>

using namespace femx;

#ifndef FEMX_GIT_COMMIT
#define FEMX_GIT_COMMIT "unknown"
#endif

#ifndef FEMX_CMAKE_BUILD_TYPE
#define FEMX_CMAKE_BUILD_TYPE ""
#endif

#ifndef FEMX_CMAKE_CXX_COMPILER
#define FEMX_CMAKE_CXX_COMPILER "unknown"
#endif

#ifndef FEMX_ENABLE_HDF5_OPTION
#define FEMX_ENABLE_HDF5_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_OPENMP_OPTION
#define FEMX_ENABLE_OPENMP_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_PETSC_OPTION
#define FEMX_ENABLE_PETSC_OPTION "unknown"
#endif

namespace
{

BuildInfo makeBuildInfo()
{
  return BuildInfo{
      {{"femx commit", FEMX_GIT_COMMIT},
       {"cmake build type", FEMX_CMAKE_BUILD_TYPE},
       {"cmake cxx compiler", FEMX_CMAKE_CXX_COMPILER},
       {"FEMX_ENABLE_HDF5", FEMX_ENABLE_HDF5_OPTION},
       {"FEMX_ENABLE_OPENMP", FEMX_ENABLE_OPENMP_OPTION},
       {"FEMX_ENABLE_PETSC", FEMX_ENABLE_PETSC_OPTION},
       {"PETSc version",
        std::to_string(PETSC_VERSION_MAJOR) + "."
            + std::to_string(PETSC_VERSION_MINOR) + "."
            + std::to_string(PETSC_VERSION_SUBMINOR)}}};
}

template <typename Fn>
double timeCollective(Fn&& fn)
{
  MPI_Barrier(PETSC_COMM_WORLD);
  const auto begin = Clock::now();
  fn();
  MPI_Barrier(PETSC_COMM_WORLD);
  return elapsedSeconds(begin, Clock::now());
}

bool isRoot()
{
  PetscMPIInt rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  return rank == 0;
}

CellRange localCellRange(Index num_cells)
{
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  const Index base  = num_cells / static_cast<Index>(size);
  const Index extra = num_cells % static_cast<Index>(size);
  const Index begin = static_cast<Index>(rank) * base
                      + std::min<Index>(rank, extra);
  const Index count = base + (rank < extra ? 1 : 0);
  return {begin, begin + count};
}

void setKspDefaults(system::KspLinearSolver& solver)
{
  auto& options         = solver.options();
  options.type          = KSPFGMRES;
  options.pc_type       = PCBJACOBI;
  options.restart       = 200;
  options.rtol          = 1.0e-8;
  options.max_its       = 5000;
  options.nonzero_guess = true;
  options.use_opts_db   = true;
}

void applyDirichletCondition(const DirichletCondition&  bc,
                             Index                      num_dofs,
                             system::PETScSystemMatrix& A,
                             system::PETScSystemVector& b,
                             system::PETScSystemVector& bc_vals)
{
  if (bc.dofs().size() != bc.values().size())
  {
    throw std::runtime_error("DirichletCondition has inconsistent data");
  }

  std::map<Index, Real> constrained;
  for (std::size_t i = 0; i < bc.dofs().size(); ++i)
  {
    const Index dof = bc.dofs()[i];
    if (dof < 0 || dof >= num_dofs)
    {
      throw std::runtime_error("Dirichlet dof is out of range");
    }

    constrained[dof] = bc.values()[i];
  }

  bc_vals.setZero();

  std::vector<Index> rows;
  rows.reserve(constrained.size());
  for (const auto& [row, value] : constrained)
  {
    rows.push_back(row);
    bc_vals.set(row, value);
  }
  bc_vals.finalize();

  A.zeroRowsColumns(rows, 1.0, bc_vals, b);
}

int run(const Params& params, bool enable_output)
{
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  if (rank == 0 && enable_output)
  {
    writeBuildInfo(params.output, makeBuildInfo());
  }

  Mesh mesh = GmshReader::read(params.mesh_file);
  auto elem = makeElem(mesh, "navier-gls-petsc");

  FESpace u_space(&mesh, elem.get(), mesh.dim());
  FESpace p_space(&mesh, elem.get());

  MixedFESpace space;
  space.addField(u_space);
  space.addField(p_space);
  space.setup();

  auto   pattern = assembly::SparsityPatternBuilder::build(space);
  Vector x(space.numDofs());
  Vector xp(space.numDofs());
  x.setZero();
  xp.setZero();

  system::PETScSystemVector b(PETSC_COMM_WORLD);
  system::PETScSystemVector x_petsc(PETSC_COMM_WORLD);
  system::PETScSystemVector bc_vals(PETSC_COMM_WORLD);
  b.resize(space.numDofs());
  x_petsc.resize(space.numDofs());
  bc_vals.resize(space.numDofs());

  system::PETScSystemMatrix A(PETSC_COMM_WORLD);
  A.resize(pattern, x_petsc);

  system::KspLinearSolver solver(PETSC_COMM_WORLD);
  setKspDefaults(solver);

  const auto cells = localCellRange(space.mesh().numElems());

  std::vector<Snapshot> snapshots;
  std::ofstream         run_log;
  if (rank == 0 && enable_output)
  {
    run_log = openRunLog(params.output);
  }
  if (rank == 0)
  {
    std::cout << "navier-gls-petsc: ranks = " << size
              << ", dofs = " << space.numDofs()
              << ", cells = " << space.mesh().numElems() << '\n';
  }

  for (Index step = 1; step <= params.time.steps; ++step)
  {
    const Real time = step * params.time.dt;

    const auto    step_start = Clock::now();
    AssemblyStats stats;
    const double  asm_time = timeCollective(
        [&]
        {
          assembleSystem(space,
                         x,
                         xp,
                         step == 1,
                         params.fluid,
                         params.time.dt,
                         cells,
                         A,
                         b,
                         stats);
        });

    if (!std::isfinite(stats.max_cfl))
    {
      throw std::runtime_error("Stopping as CFL became invalid");
    }

    const auto   bc      = makeBoundaryCondition(space, params.bcs, time);
    const double bc_time = timeCollective(
        [&]
        {
          applyDirichletCondition(bc, space.numDofs(), A, b, bc_vals);
        });

    Vector x_old = x;
    x_petsc.copyOwnedFrom(x);

    const double solve_time = timeCollective(
        [&]
        {
          solver.solve(A, b, x_petsc);
        });

    const PetscInt  its   = solver.its();
    const PetscReal rnorm = solver.rnorm();

    const double gather_time = timeCollective(
        [&]
        {
          x_petsc.copyToAll(x);
        });
    if (!isFinite(x))
    {
      throw std::runtime_error("Linear solve produced non-finite values in x");
    }
    xp = x_old;

    double out_time = 0.0;
    if (enable_output && rank == 0
        && shouldWriteOutput(step, params.time.steps, params.output))
    {
      out_time = timeBlock(
          [&]
          {
            snapshots.push_back(makeSnapshot(space, x, time));
            writeOutput(mesh, params.output, snapshots);
          });
    }

    const double total_time = elapsedSeconds(step_start, Clock::now());
    if (rank == 0)
    {
      std::ostringstream line;
      line << "step " << std::setw(7) << step << " / " << std::setw(7)
           << params.time.steps << ", t = " << std::setw(11) << time
           << ", max CFL = " << std::setw(11) << stats.max_cfl
           << ", KSP its = " << std::setw(6) << its
           << ", |r| = " << std::setw(11) << rnorm
           << ", assembly = " << std::setw(11) << asm_time << " s"
           << ", bc = " << std::setw(11) << bc_time << " s"
           << ", solve = " << std::setw(11) << solve_time << " s"
           << ", gather = " << std::setw(11) << gather_time << " s"
           << ", output = " << std::setw(11) << out_time << " s"
           << ", total = " << std::setw(11) << total_time << " s";
      std::cout << line.str() << '\n';

      if (enable_output)
      {
        run_log << "step " << std::setw(7) << step << ", t = "
                << std::setw(11) << time << ", max CFL = "
                << std::setw(11) << stats.max_cfl << ", KSP its = " << its
                << ", residual = " << rnorm << ", assembly = " << asm_time
                << ", bc = " << bc_time << ", solve = " << solve_time
                << ", gather = " << gather_time << ", output = " << out_time
                << ", total = " << total_time << '\n';
        if (shouldWriteOutput(step, params.time.steps, params.output))
        {
          run_log.flush();
        }
      }
    }
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != 0)
  {
    return 1;
  }

  int status = 0;
  try
  {
    const AppOptions options = parseAppOptions(argc, argv, true);
    if (options.help)
    {
      if (isRoot())
      {
        printUsage(std::cout,
                   "navier-gls-petsc",
                   " [PETSc options]",
                   {"Example PETSc options: -ksp_monitor -ksp_rtol 1e-8 -pc_type bjacobi"});
      }
    }
    else
    {
      Params params = loadConfig(options.config_file);
      if (options.steps)
      {
        params.time.steps = *options.steps;
      }
      status = run(params, !options.no_output);
    }
  }
  catch (const std::exception& e)
  {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0)
    {
      std::cerr << "navier-gls-petsc: " << e.what() << '\n';
    }
    status = 1;
  }

  PetscFinalize();
  return status;
}
