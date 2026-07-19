#pragma once

#include <petscksp.h>

#include <stdexcept>
#include <string>

#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/runtime/Parallel.hpp>

namespace femx::runtime
{

inline void checkPetsc(PetscErrorCode     ierr,
                       const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

class PetscSession
{
public:
  PetscSession(int&        argc,
               char**&     argv,
               const char* help = nullptr)
  {
    checkPetsc(PetscInitialize(&argc, &argv, nullptr, help),
               "PetscInitialize");
    active_ = true;
  }

  PetscSession(const PetscSession&)            = delete;
  PetscSession& operator=(const PetscSession&) = delete;
  PetscSession(PetscSession&&)                 = delete;
  PetscSession& operator=(PetscSession&&)      = delete;

  ~PetscSession()
  {
    if (active_)
    {
      PetscFinalize();
    }
  }

  PetscErrorCode finalize()
  {
    if (!active_)
    {
      return PETSC_SUCCESS;
    }
    active_ = false;
    return PetscFinalize();
  }

private:
  bool active_ = false;
};

inline PetscMPIInt commRank(MPI_Comm comm = PETSC_COMM_WORLD)
{
  PetscMPIInt rank = 0;
  checkPetsc(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
  return rank;
}

inline PetscMPIInt commSize(MPI_Comm comm = PETSC_COMM_WORLD)
{
  PetscMPIInt size = 1;
  checkPetsc(MPI_Comm_size(comm, &size), "MPI_Comm_size");
  return size;
}

inline bool isRoot(MPI_Comm comm = PETSC_COMM_WORLD)
{
  return commRank(comm) == 0;
}

inline IndexRange mpiPartitionRange(Index    count,
                                    MPI_Comm comm = PETSC_COMM_WORLD)
{
  return partitionRange(count,
                        static_cast<Index>(commRank(comm)),
                        static_cast<Index>(commSize(comm)));
}

template <typename ElementRangeOwner>
void setElemRange(ElementRangeOwner& owner,
                  Index              num_elems,
                  MPI_Comm           comm = PETSC_COMM_WORLD)
{
  const IndexRange elems = mpiPartitionRange(num_elems, comm);
  owner.setElemRange(elems.begin, elems.end);
}

inline void setPetscOptionIfMissing(const char* name,
                                    const char* value)
{
  PetscBool exists = PETSC_FALSE;
  checkPetsc(PetscOptionsHasName(nullptr, nullptr, name, &exists),
             std::string("PetscOptionsHasName(") + name + ")");
  if (exists == PETSC_TRUE)
  {
    return;
  }

  checkPetsc(PetscOptionsSetValue(nullptr, name, value),
             std::string("PetscOptionsSetValue(") + name + ")");
}

template <typename SolverParams>
void setMumpsKspOptions(linalg::KspLinearSolver& solver,
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

  setPetscOptionIfMissing("-pc_factor_mat_solver_type", "mumps");
  setPetscOptionIfMissing("-pc_factor_mat_ordering_type", "rcm");
  setPetscOptionIfMissing("-sub_pc_factor_mat_ordering_type", "rcm");
}

} // namespace femx::runtime
