#pragma once

#include <functional>

#include <femx/core/Types.hpp>
#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/solve/ReducedObjective.hpp>
#include <femx/problem/TimeObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace solve
{

using problem::TimeObjectiveFunctional;

/**
 * @brief Reduced objective for time-marching matrix equations.
 *
 * The residual convention is R_k(u_{k+1}, u_k, m) = 0 for
 * k = 0, ..., N - 1. This variant assembles next/previous state Jacobians
 * into caller-provided matrices for the adjoint solve.
 */
class TimeReducedFunctional final : public ReducedObjective
{
public:
  using ProgressCallback =
      std::function<void(const char* phase, Index step, Index total_steps)>;
  using InitialStateParamJacT =
      std::function<void(const Vector<Real>& prm,
                         const Vector<Real>& state_grad,
                         Vector<Real>&       out)>;

  TimeReducedFunctional(solve::TimeStateSolver&                  state_solver,
                        const problem::TimeMatrixResidualEquation& eq,
                        algebra::SystemMatrix&                 next_state_jac,
                        algebra::SystemMatrix&                 prev_state_jac,
                        algebra::LinearSolver&                 adjoint_solver,
                        const TimeObjectiveFunctional&        obj);

  void setProgress(ProgressCallback callback);

  void clearProgress();

  void setInitialStateParamJacT(InitialStateParamJacT callback);

  void clearInitialStateParamJacT();

  void resetTiming();

  Real assemblySeconds() const;

  Real solveSeconds() const;

  Index assemblyCalls() const;

  Index solveCalls() const;

  Index numParams() const override;

  Real value(const Vector<Real>& prm) override;

  void grad(const Vector<Real>& prm,
            Vector<Real>&       out) override;

  Real valueGrad(const Vector<Real>& prm,
                 Vector<Real>&       grad_out) override;

private:
  void checkDims() const;

  void solveFwd(const Vector<Real>&      prm,
                solve::TimeStateTrajectory& tr);

  void gradAt(const solve::TimeStateTrajectory& tr,
              const Vector<Real>&            prm,
              Vector<Real>&                  out);

  void notify(const char* phase, Index step, Index total_steps);

  static void checkSize(const Vector<Real>& value,
                        Index               expected,
                        const char*         message);

private:
  solve::TimeStateSolver&                  state_solver_;
  const problem::TimeMatrixResidualEquation& eq_;
  algebra::SystemMatrix&                 next_state_jac_;
  algebra::SystemMatrix&                 prev_state_jac_;
  algebra::LinearSolver&                 adj_solver_;
  const TimeObjectiveFunctional&        obj_;
  ProgressCallback                      callback_;
  InitialStateParamJacT                 init_param_jac_t_;
  Real                                  assembly_seconds_{0.0};
  Real                                  solve_seconds_{0.0};
  Index                                 assembly_calls_{0};
  Index                                 solve_calls_{0};
};

} // namespace solve
} // namespace femx
