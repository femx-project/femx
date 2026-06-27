#include <petscksp.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <femx/model/ns/ForwardProblem.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScMatrix.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>
#include <femx/runtime/BuildInfo.hpp>
#include <femx/runtime/Output.hpp>
#include <femx/runtime/PetscRuntime.hpp>
#include <femx/state/TimeLinearIntegrator.hpp>

using namespace std;
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

#ifndef FEMX_NS_FORWARD_APP_NAME
#define FEMX_NS_FORWARD_APP_NAME "ns-forward-petsc"
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
       {"FEMX_ENABLE_ENZYME", FEMX_ENABLE_ENZYME_OPTION},
       {"PETSc version",
        to_string(PETSC_VERSION_MAJOR) + "."
            + to_string(PETSC_VERSION_MINOR) + "."
            + to_string(PETSC_VERSION_SUBMINOR)}}};
}

void setKspOptions(KspLinearSolver& solver, const SolverParams& prm)
{
  auto& opts       = solver.opts();
  opts.restart     = 200;
  opts.rtol        = 1.0e-8;
  opts.max_its     = 5000;
  opts.use_opts_db = true;

  const PetscMPIInt comm_size = commSize(PETSC_COMM_WORLD);

  if (prm.method == "direct")
  {
    opts.type          = KSPPREONLY;
    opts.pc_type       = PCLU;
    opts.nonzero_guess = false;
    setPetscOptionIfMissing("-pc_factor_mat_solver_type", comm_size > 1 ? "mumps" : "petsc");
    setPetscOptionIfMissing("-pc_factor_mat_ordering_type", "rcm");
  }
  else
  {
    opts.type          = KSPFGMRES;
    opts.pc_type       = comm_size > 1 ? PCBJACOBI : PCILU;
    opts.nonzero_guess = true;

    if (comm_size > 1)
    {
      setPetscOptionIfMissing("-sub_pc_type", "ilu");
      setPetscOptionIfMissing("-sub_pc_factor_levels", "0");
      setPetscOptionIfMissing("-sub_pc_factor_mat_ordering_type", "rcm");
    }
    else
    {
      setPetscOptionIfMissing("-pc_factor_levels", "0");
      setPetscOptionIfMissing("-pc_factor_mat_ordering_type", "rcm");
    }
  }
}

int run(const Params& prm, bool enable_output)
{
  const PetscMPIInt rank = commRank(PETSC_COMM_WORLD);
  const PetscMPIInt size = commSize(PETSC_COMM_WORLD);

  if (rank == 0 && enable_output)
  {
    writeBuildInfo(prm.output.directory, makeBuildInfo());
  }

  ForwardProblem  fwd(prm);
  setElemRange(fwd.fem, fwd.space.mesh().numElems());

  PETScVector mat_row(PETSC_COMM_WORLD);
  mat_row.resize(fwd.space.numDofs());

  PETScMatrix A(PETSC_COMM_WORLD);
  A.resize(fwd.pettern, mat_row);

  KspLinearSolver solver(PETSC_COMM_WORLD);
  setKspOptions(solver, prm.solver);

  TimeLinearIntegrator integrator(fwd.problem, A, solver);
  integrator.setInitialState(fwd.x0);

  if (rank == 0)
  {
    cout << FEMX_NS_FORWARD_APP_NAME << ": ranks = " << size
         << ", dofs = " << fwd.space.numDofs()
         << ", elems = " << fwd.space.mesh().numElems() << '\n';
  }

  ofstream log_out;
  if (rank == 0 && enable_output)
  {
    log_out = openOutputFile(prm.output.directory, "run-info.txt");
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
  int status = 0;
  try
  {
    PetscSession petsc(argc, argv);
    setSerialOpenMp();

    try
    {
      const AppOptions opts = parseAppOptions(argc, argv, true);
      if (opts.help)
      {
        if (isRoot())
        {
          printUsage(
              cout,
              FEMX_NS_FORWARD_APP_NAME,
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
        cerr << FEMX_NS_FORWARD_APP_NAME << ": " << e.what() << '\n';
      }
      status = 1;
    }

    const PetscErrorCode ierr = petsc.finalize();
    if (ierr != PETSC_SUCCESS && status == 0)
    {
      return 1;
    }
  }
  catch (const exception& e)
  {
    cerr << FEMX_NS_FORWARD_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
  return status;
}
