#include <petscksp.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "CommandLine.hpp"
#include "Config.hpp"
#include "NavierProblem.hpp"
#include "Solve.hpp"
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/linalg/petsc/PETScLinearSystem.hpp>
#include <femx/model/navier/NavierResidual.hpp>
#include <femx/runtime/BuildInfo.hpp>
#include <femx/runtime/Output.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/TimeIntegrator.hpp>
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

#ifndef FEMX_NAVIER_APP_NAME
#define FEMX_NAVIER_APP_NAME "navier-petsc"
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

void setKspOptions(PETScLinearSolver& solver, const SolverConfig& prm)
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

int run(const Config& prm)
{
  const PetscMPIInt rank = commRank(PETSC_COMM_WORLD);
  OutputConfig      out  = prm.output;
  out.enabled            = rank == 0 && prm.output.enabled;

  if (out.enabled)
  {
    writeBuildInfo(out.directory, makeBuildInfo());
  }

  NavierProblem problem(prm);

  PETScLinearSystem system(PETSC_COMM_WORLD);

  setKspOptions(system.solver(), prm.solver);

  model::navier::HostNavierResidual     navier(problem.model());
  assembly::HostConstrainedTimeResidual res(navier, problem.controlMap());

  HostTimeIntegrator integ(res, system);
  integ.setInitialState(problem.initialState());

  std::ofstream log_out;
  if (out.enabled)
  {
    log_out = openOutputFile(out.directory, "run-info.txt");
  }

  const SolveResult result =
      solve(integ,
            problem,
            prm.time,
            out,
            rank == 0 ? &std::cout : nullptr,
            out.enabled ? &log_out : nullptr);

  if (!hasFiniteValues(result.final_state))
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
      const CommandLineOptions opts = parseCommandLine(argc, argv, true);
      if (opts.help)
      {
        if (isRoot())
        {
          printUsage(
              std::cout,
              FEMX_NAVIER_APP_NAME,
              " [PETSc options]",
              {"Example PETSc options: -ksp_monitor -ksp_rtol 1e-8 -pc_type lu"});
        }
      }
      else
      {
        const Config prm = loadConfig(opts.config_file);
        status           = run(prm);
      }
    }
    catch (const std::exception& e)
    {
      if (isRoot())
      {
        std::cerr << FEMX_NAVIER_APP_NAME << ": " << e.what() << '\n';
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
    std::cerr << FEMX_NAVIER_APP_NAME << ": " << e.what() << '\n';
    return 1;
  }
  return status;
}
