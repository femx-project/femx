#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/eq/StateSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>

namespace femx
{
namespace eq
{

struct OperatorNewtonStateSolverOptions
{
  Index max_its        = 20;
  Real  res_tol        = 1.0e-10;
  Real  step_tolerance = 0.0;
};

/** @brief Newton solver for R(u,m)=0 using an operator state Jacobian. */
class OperatorNewtonStateSolver final : public StateSolver
{
public:
  OperatorNewtonStateSolver(const ResidualEquation& eq,
                            system::LinearSolver&   lin_solver);

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
  system::LinearSolver&            lin_solver_;
  OperatorNewtonStateSolverOptions options_;
  Vector<Real>                     init_state_;
  bool                             has_init_state_{false};
};

} // namespace eq
} // namespace femx
