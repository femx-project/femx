#include <petscksp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "RunSupport.hpp"
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScMatrixOperator.hpp>
#include <femx/linalg/petsc/PETScVectorBuilder.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

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

#ifndef FEMX_ENABLE_HDF5_OPTION
#define FEMX_ENABLE_HDF5_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_OPENMP_OPTION
#define FEMX_ENABLE_OPENMP_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_PETSC_OPTION
#define FEMX_ENABLE_PETSC_OPTION "unknown"
#endif

#ifndef FEMX_ENABLE_ENZYME_OPTION
#define FEMX_ENABLE_ENZYME_OPTION "unknown"
#endif

#ifndef FEMX_NAVIER_GLS_APP_NAME
#define FEMX_NAVIER_GLS_APP_NAME "ns-gls-petsc"
#endif

namespace
{

struct ElemRange
{
  Index begin = 0;
  Index end   = 0;
};

BuildInfo makeBuildInfo()
{
  return BuildInfo{
      {{"femx commit", FEMX_GIT_COMMIT},
       {"cmake build type", FEMX_CMAKE_BUILD_TYPE},
       {"cmake cxx compiler", FEMX_CMAKE_CXX_COMPILER},
       {"FEMX_ENABLE_HDF5", FEMX_ENABLE_HDF5_OPTION},
       {"FEMX_ENABLE_OPENMP", FEMX_ENABLE_OPENMP_OPTION},
       {"FEMX_ENABLE_PETSC", FEMX_ENABLE_PETSC_OPTION},
       {"FEMX_ENABLE_ENZYME", FEMX_ENABLE_ENZYME_OPTION},
       {"PETSc version",
        to_string(PETSC_VERSION_MAJOR) + "."
            + to_string(PETSC_VERSION_MINOR) + "."
            + to_string(PETSC_VERSION_SUBMINOR)}}};
}

void checkPetsc(PetscErrorCode ierr, const string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw runtime_error(
        action + " failed with PETSc error code " + to_string(ierr));
  }
}

void setSerialOpenMp()
{
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(1);
#endif
}

bool isRoot()
{
  PetscMPIInt rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  return rank == 0;
}

ElemRange elemRange(Index num_elems)
{
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  const Index base  = num_elems / static_cast<Index>(size);
  const Index extra = num_elems % static_cast<Index>(size);
  const Index begin = static_cast<Index>(rank) * base
                      + min<Index>(rank, extra);
  const Index count = base + (rank < extra ? 1 : 0);
  return {begin, begin + count};
}

void setPetscDefaultOption(const char* name, const char* value)
{
  PetscBool      exists = PETSC_FALSE;
  PetscErrorCode ierr   = PetscOptionsHasName(nullptr, nullptr, name, &exists);
  checkPetsc(ierr, string("PetscOptionsHasName(") + name + ")");
  if (exists == PETSC_TRUE)
  {
    return;
  }
  ierr = PetscOptionsSetValue(nullptr, name, value);
  checkPetsc(ierr, string("PetscOptionsSetValue(") + name + ")");
}

void setKspOptions(KspLinearSolver& solver, const SolverParams& prm)
{
  auto& opts       = solver.opts();
  opts.restart     = 200;
  opts.rtol        = 1.0e-8;
  opts.max_its     = 5000;
  opts.use_opts_db = true;

  PetscMPIInt comm_size = 1;
  checkPetsc(MPI_Comm_size(PETSC_COMM_WORLD, &comm_size), "MPI_Comm_size");

  if (prm.method == "direct")
  {
    opts.type          = KSPPREONLY;
    opts.pc_type       = PCLU;
    opts.nonzero_guess = false;
    setPetscDefaultOption(
        "-pc_factor_mat_solver_type",
        comm_size > 1 ? "mumps" : "petsc");
    setPetscDefaultOption("-pc_factor_mat_ordering_type", "rcm");
  }
  else
  {
    opts.type          = KSPFGMRES;
    opts.pc_type       = comm_size > 1 ? PCBJACOBI : PCILU;
    opts.nonzero_guess = true;

    if (comm_size > 1)
    {
      setPetscDefaultOption("-sub_pc_type", "ilu");
      setPetscDefaultOption("-sub_pc_factor_levels", "0");
      setPetscDefaultOption("-sub_pc_factor_mat_ordering_type", "rcm");
    }
    else
    {
      setPetscDefaultOption("-pc_factor_levels", "0");
      setPetscDefaultOption("-pc_factor_mat_ordering_type", "rcm");
    }
  }
}

int run(const Params& prm, bool enable_output)
{
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  if (rank == 0 && enable_output)
  {
    writeBuildInfo(prm.output, makeBuildInfo());
  }

  ForwardProblem  fwd(prm);
  const ElemRange elems = elemRange(fwd.space.mesh().numElems());
  fwd.fem.setElemRange(elems.begin, elems.end);

  PETScVectorBuilder row_layout(PETSC_COMM_WORLD);
  row_layout.resize(fwd.space.numDofs());

  PETScMatrixOperator A(PETSC_COMM_WORLD);
  A.resize(fwd.pettern, row_layout);

  KspLinearSolver solver(PETSC_COMM_WORLD);
  setKspOptions(solver, prm.solver);

  TimeLinearIntegrator integrator(fwd.problem, A, solver);
  integrator.setInitialState(fwd.x0);

  if (rank == 0)
  {
    cout << FEMX_NAVIER_GLS_APP_NAME << ": ranks = " << size
         << ", dofs = " << fwd.space.numDofs()
         << ", elems = " << fwd.space.mesh().numElems() << '\n';
  }

  ofstream log_out;
  if (rank == 0 && enable_output)
  {
    log_out = openRunLog(prm.output);
  }

  ForwardSolveResult result;
  integrator.resetTiming();
  result = solve(integrator,
                 fwd,
                 prm.time,
                 prm.output,
                 rank == 0 && enable_output,
                 rank == 0 ? &cout : nullptr,
                 rank == 0 && enable_output ? &log_out : nullptr);

  if (!isFinite(result.final_state))
  {
    throw runtime_error("Linear solve produced non-finite values in x");
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
  setSerialOpenMp();

  int status = 0;
  try
  {
    const AppOptions opts = parseAppOptions(argc, argv, true);
    if (opts.help)
    {
      if (isRoot())
      {
        printUsage(cout,
                   FEMX_NAVIER_GLS_APP_NAME,
                   " [PETSc options]",
                   {"Example PETSc options: -ksp_monitor -ksp_rtol 1e-8 -pc_type lu"});
      }
    }
    else
    {
      Params prm = loadConfig(opts.config_file);
      if (opts.steps)
      {
        prm.time.steps = *opts.steps;
      }
      status = run(prm, !opts.no_output);
    }
  }
  catch (const exception& e)
  {
    if (isRoot())
    {
      cerr << FEMX_NAVIER_GLS_APP_NAME << ": " << e.what() << '\n';
    }
    status = 1;
  }

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS && status == 0)
  {
    return 1;
  }
  return status;
}
