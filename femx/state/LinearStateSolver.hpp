#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/problem/Linearization.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx
{
namespace state
{

/**
 * @brief State solver for affine-linear stationary residuals.
 *
 * LinearStateSolver linearizes the residual, solves the state Jacobian system,
 * and returns the stationary state associated with a parameter vector.
 */
class LinearStateSolver final : public StateSolver
{
public:
  LinearStateSolver(const problem::Residual& problem,
                    problem::Linearization&  lin,
                    linalg::LinearSolver&    lin_solver);

  Index numStates() const override;
  Index numParams() const override;
  Index numResiduals() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  const problem::Residual& problem_;
  problem::Linearization&  linearization_;
  linalg::LinearSolver&    lin_solver_;
  problem::Dimensions      dims_;
};

} // namespace state
} // namespace femx
