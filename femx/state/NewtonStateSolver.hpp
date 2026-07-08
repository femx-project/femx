#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx
{
namespace state
{

struct NewtonStateOptions
{
  Index max_its            = 20;
  Real  residual_tolerance = 1.0e-10;
  Real  step_tolerance     = 0.0;
};

/**
 * @brief State solver using Newton iterations over state::Residual.
 *
 * NewtonStateSolver linearizes at the current state, solves Newton
 * corrections, and stops according to NewtonStateOptions.
 */
class NewtonStateSolver final : public StateSolver
{
public:
  NewtonStateSolver(const state::Residual& problem,
                    state::Linearization&  lin,
                    linalg::LinearSolver&    lin_solver);

  NewtonStateOptions& opts();

  const NewtonStateOptions& opts() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numStates() const override;

  Index numParams() const override;

  Index numResiduals() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  void initializeState(Vector<Real>& state) const;

private:
  const state::Residual& problem_;
  state::Linearization&  linearization_;
  linalg::LinearSolver&    lin_solver_;
  state::Dimensions      dims_;
  NewtonStateOptions       opts_;
  Vector<Real>             init_state_;
  bool                     has_init_state_{false};
};

} // namespace state
} // namespace femx
