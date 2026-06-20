#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace solve
{

/** @brief Solver for the adjoint equation R_u(u,m)^T lambda = rhs. */
class AdjointSolver
{
public:
  virtual ~AdjointSolver() = default;

  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;
  virtual Index numRes() const    = 0;

  virtual void solve(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& rhs,
                     Vector<Real>&       adjoint) = 0;
};

} // namespace solve
} // namespace femx
