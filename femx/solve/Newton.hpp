#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/Residual.hpp>

namespace femx
{
namespace solve
{

struct NewtonOptions
{
  Index max_its            = 20;
  Real  residual_tolerance = 1.0e-10;
  Real  step_tolerance     = 0.0;
};

/** @brief Newton solver for problem::Residual using caller-owned linearization. */
class Newton final
{
public:
  Newton(const problem::Residual& problem,
         problem::Linearization&  linearization,
         linalg::LinearSolver&    linear_solver);

  NewtonOptions& options();

  const NewtonOptions& options() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numStates() const;

  Index numParams() const;

  Index numResiduals() const;

  problem::Linearization& linearization();

  const problem::Linearization& linearization() const;

  void solve(const Vector<Real>& prm, Vector<Real>& state);

private:
  void initializeState(Vector<Real>& state) const;

  static void resize(Vector<Real>& out, Index size);

  static Real squaredNorm(const Vector<Real>& x);

private:
  const problem::Residual& problem_;
  problem::Linearization&  linearization_;
  linalg::LinearSolver&    linear_solver_;
  problem::Dimensions      dims_;
  NewtonOptions            options_;
  Vector<Real>             init_state_;
  bool                     has_init_state_{false};
};

} // namespace solve
} // namespace femx
