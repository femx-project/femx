#pragma once

#include <femx/linalg/Vector.hpp>
#include <femx/linalg/LinearOperator.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Solver for A x = b and A^T x = b.
 *
 * Backends such as PETSc and ReSolve implement this interface so state and
 * adjoint solvers can be written independently of the linear algebra package.
 */
class LinearSolver
{
public:
  virtual ~LinearSolver() = default;

  /** @brief Solve op out = rhs. */
  virtual void solve(const LinearOperator& op,
                     const Vector<Real>&   rhs,
                     Vector<Real>&         out) = 0;

  /** @brief Solve op^T out = rhs. */
  virtual void solveT(const LinearOperator& op,
                      const Vector<Real>&   rhs,
                      Vector<Real>&         out) = 0;
};

} // namespace linalg
} // namespace femx
