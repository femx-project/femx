#pragma once

#include <femx/common/Types.hpp>
#include <femx/state/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace linalg
{
class LinearSolver;
} // namespace linalg

namespace state
{

class Linearization;

/**
 * @brief State solver for affine-linear stationary residuals.
 *
 * LinearStateSolver linearizes the residual, solves the state Jacobian system,
 * and returns the stationary state associated with a parameter vector.
 */
class LinearStateSolver final : public StateSolver
{
public:
  LinearStateSolver(const state::Residual& problem,
                    state::Linearization&  lin,
                    linalg::LinearSolver&  lin_solver);

  Index numStates() const override;
  Index numParams() const override;
  Index numResiduals() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  const state::Residual& problem_;
  state::Linearization&  linearization_;
  linalg::LinearSolver&  lin_solver_;
  state::Dimensions      dims_;
};

} // namespace state
} // namespace femx
