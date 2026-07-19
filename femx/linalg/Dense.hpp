#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/** @brief Small row-major Host matrix storage. */
class DenseMatrix final
{
public:
  DenseMatrix() = default;
  DenseMatrix(Index rows, Index cols);

  void resize(Index rows, Index cols);
  void setZero();

  Index rows() const;
  Index cols() const;

  Index size() const;

  Real& operator()(Index i, Index j);
  Real  operator()(Index i, Index j) const;

  Real*       data();
  const Real* data() const;

  HostMatrixView<Real>       view();
  HostMatrixView<const Real> view() const;

private:
  Index      rows_{0};
  Index      cols_{0};
  HostVector vals_;
};

/** @brief Compute `y = alpha * mat * x + beta * y` on Host. */
void apply(HostMatrixView<const Real> mat,
           HostConstVectorView        x,
           HostVectorView             y,
           CpuContext&                ctx,
           Real                       alpha = 1.0,
           Real                       beta  = 0.0);

/** @brief Compute `y = alpha * mat^T * x + beta * y` on Host. */
void applyT(HostMatrixView<const Real> mat,
            HostConstVectorView        x,
            HostVectorView             y,
            CpuContext&                ctx,
            Real                       alpha = 1.0,
            Real                       beta  = 0.0);

/** @brief Compute `y = alpha * mat * x + beta * y` with cuBLAS. */
void apply(DeviceMatrixView<const Real> mat,
           DeviceConstVectorView        x,
           DeviceVectorView             y,
           CudaContext&                 ctx,
           Real                         alpha = 1.0,
           Real                         beta  = 0.0);

/** @brief Compute `y = alpha * mat^T * x + beta * y` with cuBLAS. */
void applyT(DeviceMatrixView<const Real> mat,
            DeviceConstVectorView        x,
            DeviceVectorView             y,
            CudaContext&                 ctx,
            Real                         alpha = 1.0,
            Real                         beta  = 0.0);

namespace linalg
{

/** @brief Dense fallback solver for small problems and tests. */
class DenseLinearSolver final : public HostCsrLinearSolver
{
public:
  explicit DenseLinearSolver(Real pivot_tolerance = 1.0e-14);

  void solve(const HostCsrMatrix& mat,
             const HostVector&    rhs,
             HostVector&          out,
             CpuContext&          ctx) override;

  void solveT(const HostCsrMatrix& mat,
              const HostVector&    rhs,
              HostVector&          out,
              CpuContext&          ctx) override;

private:
  void sample(const HostCsrMatrix& mat,
              bool                 tr,
              HostVector&          dense) const;

  void solveDense(HostVector        mat,
                  const HostVector& rhs,
                  HostVector&       out,
                  Index             size) const;

  static Index entry(Index row, Index col, Index size);

  Real pivot_tolerance_{1.0e-14};
};

} // namespace linalg
} // namespace femx
