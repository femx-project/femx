#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/eq/TimeMatrixResidualEquation.hpp>
#include <femx/eq/TimeStateSolver.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Reduced objective for time-marching matrix equations.
 *
 * The residual convention is R_k(u_{k+1}, u_k, m) = 0 for
 * k = 0, ..., N - 1. This variant assembles next/previous state Jacobians
 * into caller-provided matrices for the adjoint solve.
 */
class TimeReducedFunctional final : public ReducedFunctional
{
public:
  using ProgressCallback =
      std::function<void(const char* phase, Index step, Index total_steps)>;
  using InitialStateParamJacT =
      std::function<void(const Vector<Real>& prm,
                         const Vector<Real>& state_grad,
                         Vector<Real>&       out)>;

  TimeReducedFunctional(eq::TimeStateSolver&                  state_solver,
                        const eq::TimeMatrixResidualEquation& eq,
                        system::SystemMatrix&                 next_state_jac,
                        system::SystemMatrix&                 prev_state_jac,
                        system::LinearSolver&                 adjoint_solver,
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
                eq::TimeStateTrajectory& tr);

  void gradAt(const eq::TimeStateTrajectory& tr,
              const Vector<Real>&            prm,
              Vector<Real>&                  out);

  void notify(const char* phase, Index step, Index total_steps);

  static void checkSize(const Vector<Real>& value,
                        Index               expected,
                        const char*         message);

private:
  eq::TimeStateSolver&                  state_solver_;
  const eq::TimeMatrixResidualEquation& eq_;
  system::SystemMatrix&                 next_state_jac_;
  system::SystemMatrix&                 prev_state_jac_;
  system::LinearSolver&                 adj_solver_;
  const TimeObjectiveFunctional&        obj_;
  ProgressCallback                      callback_;
  InitialStateParamJacT                 init_param_jac_t_;
  Real                                  assembly_seconds_{0.0};
  Real                                  solve_seconds_{0.0};
  Index                                 assembly_calls_{0};
  Index                                 solve_calls_{0};
};

} // namespace inverse
} // namespace femx
