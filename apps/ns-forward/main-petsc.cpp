#include <petscksp.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>
#include <femx/model/ns/ForwardProblem.hpp>
#include <femx/runtime/BuildInfo.hpp>
#include <femx/runtime/Output.hpp>
#include <femx/runtime/PETScRuntime.hpp>
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
        std::to_string(PETSC_VERSION_MAJOR) + "."
            + std::to_string(PETSC_VERSION_MINOR) + "."
            + std::to_string(PETSC_VERSION_SUBMINOR)}}};
}

void setKspOptions(KspLinearSolver& solver, const SolverParams& prm)
{
  auto& opts       = solver.opts();
  opts.restart     = prm.restart;
  opts.rtol        = prm.relative_tolerance;
  opts.max_its     = prm.max_itrs;
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

int run(const Params& prm)
{
  const PetscMPIInt rank   = commRank(PETSC_COMM_WORLD);
  OutputParams      output = prm.output;
  output.enabled           = rank == 0 && prm.output.enabled;

  if (output.enabled)
  {
    writeBuildInfo(output.directory, makeBuildInfo());
  }

  ForwardProblem fwd(prm);
  setElemRange(fwd.model.residual(), fwd.model.mesh().numElems());

  PETScVector mat_row(PETSC_COMM_WORLD);
  mat_row.resize(fwd.model.numStates());

  PETScOperator A(PETSC_COMM_WORLD);
  A.resize(fwd.model.map().graph(), mat_row);

  KspLinearSolver solver(PETSC_COMM_WORLD);
  setKspOptions(solver, prm.solver);

  TimeLinearIntegrator integ(fwd.problem, A, solver);
  integ.setInitialState(fwd.x0);

  std::ofstream log_out;
  if (output.enabled)
  {
    log_out = openOutputFile(output.directory, "run-info.txt");
  }

  ForwardSolveResult result;
  integ.resetTiming();
  result = solve(integ,
                 fwd,
                 prm.time,
                 output,
                 rank == 0 ? &std::cout : nullptr,
                 output.enabled ? &log_out : nullptr);

  if (!isFinite(result.final_state))
  {
    throw std::runtime_error("Linear solve produced non-finite values in x");
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
              std::cout,
              FEMX_NS_FORWARD_APP_NAME,
              " [PETSc options]",
              {"Example PETSc options: -ksp_monitor -ksp_rtol 1e-8 -pc_type lu"});
        }
      }
      else
      {
        Params prm = loadConfig(opts.config_file);
        status     = run(prm);
      }
    }
    catch (const std::exception& e)
    {
      if (isRoot())
      {
        std::cerr << FEMX_NS_FORWARD_APP_NAME << ": " << e.what() << '\n';
      }
      status = 1;
    }

    const PetscErrorCode ierr = petsc.finalize();
    if (ierr != PETSC_SUCCESS && status == 0)
    {
      return 1;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_NS_FORWARD_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
  return status;
}
