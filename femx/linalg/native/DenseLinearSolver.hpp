#pragma once

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx::linalg
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
              bool                 transpose,
              DenseMatrix&         dense) const;

  void solveDense(DenseMatrix       mat,
                  const HostVector& rhs,
                  HostVector&       out,
                  CpuContext&       ctx) const;

  Real pivot_tolerance_{1.0e-14};
};

} // namespace femx::linalg
