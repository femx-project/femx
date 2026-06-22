#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/Residual.hpp>
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

/** @brief State solver using Newton iterations over problem::Residual. */
class NewtonStateSolver final : public StateSolver
{
public:
  NewtonStateSolver(const problem::Residual& problem,
                    problem::Linearization&  lin,
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

  static Real squaredNorm(const Vector<Real>& x);

private:
  const problem::Residual& problem_;
  problem::Linearization&  linearization_;
  linalg::LinearSolver&    lin_solver_;
  problem::Dimensions      dims_;
  NewtonStateOptions       opts_;
  Vector<Real>             init_state_;
  bool                     has_init_state_{false};
};

} // namespace state
} // namespace femx
