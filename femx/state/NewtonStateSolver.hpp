#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx
{
namespace linalg
{
class LinearSolver;
} // namespace linalg

namespace state
{

class Linearization;

struct NewtonStateOptions
{
  Index max_its        = 20;      ///< Maximum Newton iterations.
  Real  res_tol        = 1.0e-10; ///< Stop when residual norm is below this.
  Real  step_tolerance = 0.0;     ///< Stop when step norm is below this.
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
                    linalg::LinearSolver&  lin_solver);

  NewtonStateOptions& opts();

  const NewtonStateOptions& opts() const;

  void setInitialState(const HostVector& state);

  void clearInitialState();

  Index numStates() const override;

  Index numParams() const override;

  Index numRes() const override;

  void solve(const HostVector& prm, HostVector& state) override;

private:
  void initializeState(HostVector& state) const;

private:
  const state::Residual& problem_;
  state::Linearization&  linearization_;
  linalg::LinearSolver&  lin_solver_;
  state::Dimensions      dims_;
  NewtonStateOptions     opts_;
  HostVector             init_state_;
  bool                   has_init_state_{false};
};

} // namespace state
} // namespace femx
