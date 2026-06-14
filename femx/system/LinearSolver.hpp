#pragma once

#include <femx/system/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{

/** @brief Solver for A x = b and A^T x = b with a matrix-free operator. */
class LinearSolver
{
public:
  virtual ~LinearSolver() = default;

  virtual void solve(const LinearOperator& op,
                     const Vector&         rhs,
                     Vector&               out) = 0;

  virtual void solveT(const LinearOperator& op,
                      const Vector&         rhs,
                      Vector&               out) = 0;
};

} // namespace system
} // namespace femx
