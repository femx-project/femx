#pragma once

#include <petscvec.h>

#include <algorithm>

#include <femx/linalg/Backend.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
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

  void sync() const noexcept
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
  using Pattern   = HostCsrPattern;
  using Ctx       = PetscContext;
};

static_assert(is_backend_v<PetscBackend>,
              "PetscBackend does not satisfy the backend contract");

template <>
class VectorHandler<PetscBackend> final
{
public:
  explicit VectorHandler(PetscContext&) noexcept
  {
  }

  void copy(HostConstVectorView src, HostVector& dst) const
  {
    dst = src;
  }

  void copy(HostConstVectorView src, HostVectorView dst) const
  {
    require(src.size() == dst.size(),
            "PETSc backend vector copy requires equal sizes");
    if (src.empty() || src.data() == dst.data())
    {
      return;
    }
    require(!femx::detail::overlaps(src, dst),
            "PETSc backend vector copy does not support partial overlap");
    std::copy(src.begin(), src.end(), dst.begin());
  }

  void zero(HostVectorView vals) const
  {
    std::fill(vals.begin(), vals.end(), Real{});
  }

  void axpby(Real                a,
             HostConstVectorView x,
             Real                b,
             HostVectorView      y) const
  {
    require(x.size() == y.size(),
            "PETSc backend axpby requires equal vector sizes");
    require(x.data() == y.data() || !femx::detail::overlaps(x, y),
            "PETSc backend axpby does not support partial overlap");
    for (Index i = 0; i < x.size(); ++i)
    {
      y[i] = a * x[i] + b * y[i];
    }
  }

  Real squaredNorm(HostConstVectorView x) const
  {
    return dot(x, x);
  }

  Real dot(HostConstVectorView x, HostConstVectorView y) const
  {
    require(x.size() == y.size(),
            "PETSc backend dot requires equal vector sizes");
    Real val = 0.0;
    for (Index i = 0; i < x.size(); ++i)
    {
      val += x[i] * y[i];
    }
    return val;
  }

  void gather(HostConstVectorView src,
              HostConstIndexView  indices,
              HostVectorView      dst) const
  {
    require(indices.size() == dst.size(),
            "PETSc backend gather output size mismatch");
    require(!femx::detail::overlaps(src, dst),
            "PETSc backend gather does not support aliased vectors");
    for (Index i = 0; i < indices.size(); ++i)
    {
      require(indices[i] >= 0 && indices[i] < src.size(),
              "PETSc backend gather index is out of range");
      dst[i] = src[indices[i]];
    }
  }

  void scatter(HostConstVectorView src,
               HostConstIndexView  indices,
               HostVectorView      dst) const
  {
    require(src.size() == indices.size(),
            "PETSc backend scatter input size mismatch");
    require(!femx::detail::overlaps(src, dst),
            "PETSc backend scatter does not support aliased vectors");
    for (Index i = 0; i < indices.size(); ++i)
    {
      require(indices[i] >= 0 && indices[i] < dst.size(),
              "PETSc backend scatter index is out of range");
      dst[indices[i]] = src[i];
    }
  }

  template <class T>
  void resizeOrZero(Vector<MemorySpace::Host, T>& out, Index size) const
  {
    if (out.size() != size)
    {
      out.resize(size);
    }
    else
    {
      std::fill(out.begin(), out.end(), T{});
    }
  }
};

template <>
class MatrixHandler<PetscBackend> final
{
public:
  explicit MatrixHandler(PetscContext&) noexcept
  {
  }

  void matvec(const PETScOperator& mat,
              HostConstVectorView  dir,
              HostVector&          out) const
  {
    mat.apply(dir, out);
  }

  void matvecT(const PETScOperator& mat,
               HostConstVectorView  dir,
               HostVector&          out) const
  {
    mat.applyT(dir, out);
  }

  void zero(PETScOperator& mat) const
  {
    mat.setZero();
  }

  void copy(const HostCsrMatrix& src, PETScOperator& dst) const
  {
    require(dst.rows() == src.rows() && dst.cols() == src.cols(),
            "PETSc matrix copy requires equal dimensions");
    zero(dst);
    for (Index row = 0; row < src.rows(); ++row)
    {
      for (Index k = src.rowPtrData()[row];
           k < src.rowPtrData()[row + 1];
           ++k)
      {
        dst.set(row, src.colIndData()[k], src.valsData()[k]);
      }
    }
  }

  void finalize(PETScOperator& mat) const
  {
    mat.finalize();
  }
};

} // namespace femx::linalg
