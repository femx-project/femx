#pragma once

#include <petscvec.h>

#include <femx/linalg/Backend.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>

namespace femx::linalg
{

/// @cond INTERNAL
namespace detail
{
void check(PetscErrorCode ierr, const char* op);
void checkMPI(int ierr, const char* op);
void checkInit();

PetscErrorCode copyFromPETSc(Vec src, HostVector& dst);
PetscErrorCode copyToPETSc(HostConstVectorView src, Vec dst);
} // namespace detail

/// @endcond

/** @brief Explicit PETSc execution context. */
struct PetscContext
{
  MPI_Comm comm{PETSC_COMM_SELF};

  void synchronize() const noexcept
  {
  }
};

/** @brief PETSc matrix/vector execution independent of MemorySpace. */
struct PetscBackend
{
  static constexpr MemorySpace space = MemorySpace::Host;

  using Vec       = HostVector;
  using VecView   = HostVectorView;
  using ConstView = HostConstVectorView;
  using Mat       = PETScOperator;
  using Graph     = HostCsrGraph;
  using Ctx       = PetscContext;
};

static_assert(is_backend_v<PetscBackend>,
              "PetscBackend does not satisfy the backend contract");

inline void copy(HostConstVectorView src,
                 HostVector&         dst,
                 PetscContext&)
{
  dst = src;
}

inline void copy(HostConstVectorView src,
                 HostVectorView      dst,
                 PetscContext&)
{
  CpuContext ctx;
  femx::copy(src, dst, ctx);
}

inline void apply(const PETScOperator& mat,
                  HostConstVectorView  dir,
                  HostVector&          out,
                  PetscContext&)
{
  mat.apply(dir, out);
}

inline void applyT(const PETScOperator& mat,
                   HostConstVectorView  dir,
                   HostVector&          out,
                   PetscContext&)
{
  mat.applyT(dir, out);
}

inline void axpby(Real                a,
                  HostConstVectorView x,
                  Real                b,
                  HostVectorView      y,
                  PetscContext&)
{
  CpuContext ctx;
  femx::axpby(a, x, b, y, ctx);
}

inline Real squaredNorm(HostConstVectorView x, PetscContext&)
{
  CpuContext ctx;
  return femx::squaredNorm(x, ctx);
}

inline void finalize(PETScOperator& mat, PetscContext&)
{
  mat.finalize();
}

} // namespace femx::linalg
