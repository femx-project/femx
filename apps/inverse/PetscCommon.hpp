#pragma once

#include <petscksp.h>

#include <algorithm>
#include <string>

#include "Common.hpp"
#include "PetscError.hpp"
#include <femx/linalg/petsc/KspLinearSolver.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace femx::inverse
{

inline void setSerialOpenMp()
{
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(1);
#endif
}

inline void setPetscOpt(const char* name,
                        const char* value)
{
  PetscBool      exists = PETSC_FALSE;
  PetscErrorCode ierr =
      PetscOptionsHasName(nullptr, nullptr, name, &exists);
  checkPetsc(ierr, std::string("PetscOptionsHasName(") + name + ")");
  if (exists == PETSC_TRUE)
  {
    return;
  }

  ierr = PetscOptionsSetValue(nullptr, name, value);
  checkPetsc(ierr, std::string("PetscOptionsSetValue(") + name + ")");
}

template <typename SolverParams>
void setKspOptions(linalg::KspLinearSolver& solver,
                   const SolverParams&      prm)
{
  auto& opts       = solver.opts();
  opts.restart     = 200;
  opts.rtol        = 1.0e-8;
  opts.max_its     = 5000;
  opts.use_opts_db = true;

  if (prm.method == "direct")
  {
    opts.type          = KSPPREONLY;
    opts.pc_type       = PCLU;
    opts.nonzero_guess = false;
  }
  else
  {
    opts.type          = KSPFGMRES;
    opts.pc_type       = PCLU;
    opts.nonzero_guess = true;
  }

  setPetscOpt("-pc_factor_mat_solver_type", "mumps");
  setPetscOpt("-pc_factor_mat_ordering_type", "rcm");
  setPetscOpt("-sub_pc_factor_mat_ordering_type", "rcm");
}

inline ElemRange mpiElemRange(Index    num_elems,
                              MPI_Comm comm = PETSC_COMM_WORLD)
{
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const Index base  = num_elems / static_cast<Index>(size);
  const Index extra = num_elems % static_cast<Index>(size);
  const Index begin = static_cast<Index>(rank) * base
                      + std::min<Index>(rank, extra);
  const Index count = base + (rank < extra ? 1 : 0);
  return {begin, begin + count};
}

} // namespace femx::inverse
