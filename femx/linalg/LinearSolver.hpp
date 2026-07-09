#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace linalg
{

class LinearOperator;

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
