#pragma once

#include <petscvec.h>

#include <femx/linalg/Backend.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/petsc/PETScOperator.hpp>

namespace femx::linalg
{

/// @cond INTERNAL
namespace detail
{
void check(PetscErrorCode ierr, const char* op);

void checkMPI(int ierr, const char* op);

void checkInit();

PetscErrorCode copyFromPETSc(Vec src, HostVector<Real>& dst);

PetscErrorCode copyToPETSc(HostVectorView<const Real> src, Vec dst);
} // namespace detail

/// @endcond

/** @brief Provide PETSc matrix and vector execution. */
struct PetscBackend
{
  static constexpr MemorySpace space = MemorySpace::Host; ///< Vector storage memory space.

  using Vec       = HostVector<Real>;
  using VecView   = HostVectorView<Real>;
  using ConstView = HostVectorView<const Real>;
  using Mat       = PETScOperator;
  using Pattern   = HostCsrPattern;
};

static_assert(is_backend_v<PetscBackend>,
              "PetscBackend does not satisfy the backend contract");

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
  explicit MatrixHandler(Context<MemorySpace::Host>& ctx) noexcept
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
  void matvec(const PETScOperator&       mat,
              HostVectorView<const Real> dir,
              HostVector<Real>&          out) const
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
  void matvecT(const PETScOperator&       mat,
               HostVectorView<const Real> dir,
               HostVector<Real>&          out) const
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
