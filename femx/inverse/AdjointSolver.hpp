#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Solver for the adjoint equation R_u(u,m)^T lambda = rhs. */
class AdjointSolver
{
public:
  virtual ~AdjointSolver() = default;

  virtual Index numStates() const    = 0;
  virtual Index numParams() const    = 0;
  virtual Index numRes() const = 0;

  virtual void solve(const Vector& state,
                     const Vector& params,
                     const Vector& rhs,
                     Vector&       adjoint) = 0;
};

} // namespace inverse
} // namespace femx
