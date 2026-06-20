#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/ResidualEquation.hpp>
#include <femx/solve/StateSolver.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>

namespace femx
{
namespace solve
{

using problem::ResidualEquation;

struct OperatorNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/** @brief Compatibility solver. Prefer solve::Newton for new code. */
class OperatorNewtonStateSolver final : public StateSolver
{
public:
  OperatorNewtonStateSolver(const ResidualEquation& eq,
                            algebra::LinearSolver&   lin_solver);

  OperatorNewtonStateSolverOptions& options();

  const OperatorNewtonStateSolverOptions& options() const;

  void setInitialState(const Vector<Real>& state);

  void clearInitialState();

  Index numStates() const override;

  Index numParams() const override;

  void solve(const Vector<Real>& prm, Vector<Real>& state) override;

private:
  void initializeState(Vector<Real>& state) const;

  static void resize(Vector<Real>& out, Index size);

private:
  const ResidualEquation&          eq_;
  algebra::LinearSolver&            lin_solver_;
  OperatorNewtonStateSolverOptions options_;
  Vector<Real>                     init_state_;
  bool                             has_init_state_{false};
};

} // namespace solve
} // namespace femx
