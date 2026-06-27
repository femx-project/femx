#pragma once

#include <femx/linalg/operator/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/** @brief Solver for A x = b and A^T x = b with a linear operator. */
class LinearSolver
{
public:
  virtual ~LinearSolver() = default;

  virtual void solve(const LinearOperator& op,
                     const Vector<Real>&   rhs,
                     Vector<Real>&         out) = 0;

  virtual void solveT(const LinearOperator& op,
                      const Vector<Real>&   rhs,
                      Vector<Real>&         out) = 0;
};

} // namespace linalg
} // namespace femx
