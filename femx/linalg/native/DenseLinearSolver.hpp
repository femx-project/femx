#pragma once

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx::linalg
{

/** @brief Solve small Host CSR systems through dense factorization. */
class DenseLinearSolver final : public HostCsrLinearSolver
{
public:
  /**
   * @brief Construct a dense solver with a pivot tolerance.
   *
   * @param[in] pivot_tolerance - Minimum accepted pivot magnitude.
   * @throws std::runtime_error - If `pivot_tolerance` is negative.
   */
  explicit DenseLinearSolver(Real pivot_tolerance = 1.0e-14);

  /**
   * @brief Solve a Host CSR system through dense factorization.
   *
   * @param[in] mat - Square system matrix.
   * @param[in] rhs - Right-hand side vector.
   * @param[out] out - Solution vector.
   * @param[in] ctx - CPU execution context.
   * @throws std::runtime_error - If dimensions are inconsistent or the matrix
   * is singular within the configured tolerance.
   */
  void solve(const HostCsrMatrix& mat,
             const HostVector&    rhs,
             HostVector&          out,
             CpuContext&          ctx) override;

  /**
   * @brief Solve a transposed Host CSR system through dense factorization.
   *
   * @param[in] mat - Square system matrix.
   * @param[in] rhs - Right-hand side vector.
   * @param[out] out - Solution vector.
   * @param[in] ctx - CPU execution context.
   * @throws std::runtime_error - If dimensions are inconsistent or the matrix
   * is singular within the configured tolerance.
   */
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

  Real pivot_tolerance_{1.0e-14}; ///< Minimum accepted pivot magnitude.
};

} // namespace femx::linalg
