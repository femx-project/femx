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

/** @brief Provide an explicit PETSc execution context. */
struct PetscContext
{
  MPI_Comm comm{PETSC_COMM_SELF}; ///< Communicator used by PETSc operations.

  /** @brief Complete context synchronization immediately. */
  void sync() const noexcept
  {
  }
};

/** @brief Provide PETSc matrix and vector execution. */
struct PetscBackend
{
  static constexpr MemorySpace space = MemorySpace::Host; ///< Vector storage memory space.

  using Vec       = HostVector;
  using VecView   = HostVectorView;
  using ConstView = HostConstVectorView;
  using Mat       = PETScOperator;
  using Pattern   = HostCsrPattern;
  using Ctx       = PetscContext;
};

static_assert(is_backend_v<PetscBackend>,
              "PetscBackend does not satisfy the backend contract");

/** @brief Provide Host vector operations for the PETSc backend. */
template <>
class VectorHandler<PetscBackend> final
{
public:
  /**
   * @brief Construct vector operations for a PETSc context.
   *
   * @param[in] ctx - PETSc execution context.
   */
  explicit VectorHandler(PetscContext& ctx) noexcept
  {
    static_cast<void>(ctx);
  }

  /**
   * @brief Replace a Host vector by copying a view.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination vector.
   * @throws std::runtime_error - If the view size is negative.
   */
  void copy(HostConstVectorView src, HostVector& dst) const
  {
    dst = src;
  }

  /**
   * @brief Copy between same-sized Host views.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error - If sizes differ or views partially overlap.
   */
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

  /**
   * @brief Set every value to zero.
   *
   * @param[out] vals - Values to clear.
   */
  void zero(HostVectorView vals) const
  {
    std::fill(vals.begin(), vals.end(), Real{});
  }

  /**
   * @brief Compute `y = a * x + b * y`.
   *
   * @param[in] a - Input-vector scale.
   * @param[in] x - Input vector.
   * @param[in] b - Existing-output scale.
   * @param[in,out] y - Output vector.
   * @throws std::runtime_error - If sizes or storage overlap are invalid.
   */
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

  /**
   * @brief Compute the squared Euclidean norm of a vector.
   *
   * @param[in] x - Input vector.
   * @return Squared Euclidean norm of `x`.
   */
  Real squaredNorm(HostConstVectorView x) const
  {
    return dot(x, x);
  }

  /**
   * @brief Compute the dot product of two vectors.
   *
   * @param[in] x - First input vector.
   * @param[in] y - Second input vector.
   * @return Dot product of `x` and `y`.
   * @throws std::runtime_error - If vector sizes differ.
   */
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

  /**
   * @brief Gather indexed source values into a contiguous destination.
   *
   * @param[in] src - Source values.
   * @param[in] indices - Source indices in destination order.
   * @param[out] dst - Contiguous destination values.
   * @throws std::runtime_error - If sizes, indices, or aliasing are invalid.
   */
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

  /**
   * @brief Scatter contiguous source values to indexed destinations.
   *
   * @param[in] src - Contiguous source values.
   * @param[in] indices - Destination indices in source order.
   * @param[out] dst - Indexed destination values.
   * @throws std::runtime_error - If sizes, indices, or aliasing are invalid.
   */
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

  /**
   * @brief Resize a Host vector if needed and set every value to zero.
   *
   * @param[in,out] out - Vector to resize or clear.
   * @param[in] size - Required vector size.
   * @throws std::runtime_error - If `size` is negative.
   */
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

/** @brief Provide PETSc matrix operations. */
template <>
class MatrixHandler<PetscBackend> final
{
public:
  /**
   * @brief Construct matrix operations for a PETSc context.
   *
   * @param[in] ctx - PETSc execution context.
   */
  explicit MatrixHandler(PetscContext& ctx) noexcept
  {
    static_cast<void>(ctx);
  }

  /**
   * @brief Compute `out = mat * dir`.
   *
   * @param[in] mat - PETSc matrix.
   * @param[in] dir - Input vector.
   * @param[out] out - Result vector.
   * @throws std::runtime_error - If inputs are invalid or PETSc reports an
   * error.
   */
  void matvec(const PETScOperator& mat,
              HostConstVectorView  dir,
              HostVector&          out) const
  {
    mat.apply(dir, out);
  }

  /**
   * @brief Compute `out = mat^T * dir`.
   *
   * @param[in] mat - PETSc matrix.
   * @param[in] dir - Input vector.
   * @param[out] out - Result vector.
   * @throws std::runtime_error - If inputs are invalid or PETSc reports an
   * error.
   */
  void matvecT(const PETScOperator& mat,
               HostConstVectorView  dir,
               HostVector&          out) const
  {
    mat.applyT(dir, out);
  }

  /**
   * @brief Set every numeric matrix entry to zero.
   *
   * @param[in,out] mat - Matrix whose values are cleared.
   * @throws std::runtime_error - If PETSc reports an error.
   */
  void zero(PETScOperator& mat) const
  {
    mat.setZero();
  }

  /**
   * @brief Copy Host CSR values into a compatible PETSc matrix.
   *
   * @param[in] src - Source Host CSR matrix.
   * @param[out] dst - Destination PETSc matrix.
   * @throws std::runtime_error - If dimensions differ or PETSc reports an
   * error.
   */
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

  /**
   * @brief Complete PETSc matrix assembly.
   *
   * @param[in,out] mat - Matrix to finalize.
   * @throws std::runtime_error - If PETSc reports an error.
   */
  void finalize(PETScOperator& mat) const
  {
    mat.finalize();
  }
};

} // namespace femx::linalg
