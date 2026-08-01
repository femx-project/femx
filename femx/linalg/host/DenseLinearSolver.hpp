#pragma once

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx::linalg
{

/**
 * @brief Solve small Host CSR systems through dense factorization.
 */
class DenseLinearSolver final : public LinearSolver<MemorySpace::Host>
{
public:
  /**
   * @brief Construct a dense solver with a pivot tolerance.
   *
   * @param[in] pivot_tolerance - Minimum accepted pivot magnitude.
   * @throws std::runtime_error If validation fails.
   */
  explicit DenseLinearSolver(Real pivot_tolerance = 1.0e-14);

  /**
   * @copydoc LinearSolver::solve()
   *
   * @details Uses dense factorization.
   * @throws std::runtime_error If validation fails.
   */
  void solve(const HostCsrMatrix&        mat,
             const HostVector<Real>&     rhs,
             HostVector<Real>&           x,
             Context<MemorySpace::Host>& ctx) override;

  /**
   * @copydoc LinearSolver::solveT()
   *
   * @details Uses dense factorization.
   * @throws std::runtime_error If validation fails.
   */
  void solveT(const HostCsrMatrix&        mat,
              const HostVector<Real>&     rhs,
              HostVector<Real>&           x,
              Context<MemorySpace::Host>& ctx) override;

private:
  void copyToDense(const HostCsrMatrix& mat,
                   bool                 transpose,
                   DenseMatrix&         dense) const;

  void solveDense(DenseMatrix                 mat,
                  const HostVector<Real>&     rhs,
                  HostVector<Real>&           x,
                  Context<MemorySpace::Host>& ctx) const;

  Real pivot_tol_; ///< Minimum accepted pivot magnitude.
};

} // namespace femx::linalg
