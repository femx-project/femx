#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx
{
namespace solve
{

class TimeReducedFunctional final
{
public:
  using ProgressCallback = std::function<void(const char* phase, Index step, Index total_steps)>;

  using InitialStateParamJacT =
      std::function<void(const Vector<Real>& prm,
                         const Vector<Real>& state_grad,
                         Vector<Real>&       out)>;

  TimeReducedFunctional(TimeStateSolver&              state_solver,
                        const problem::TimeResidual&  eq,
                        linalg::MatrixOperator&       next_state_jac,
                        linalg::MatrixOperator&       prev_state_jac,
                        linalg::LinearSolver&         adjoint_solver,
                        const problem::TimeObjective& obj);

  void setProgress(ProgressCallback callback);
  void clearProgress();

  void setInitialStateParamJacT(InitialStateParamJacT callback);
  void clearInitialStateParamJacT();

  void  resetTiming();
  Real  assemblySeconds() const;
  Real  solveSeconds() const;
  Index assemblyCalls() const;
  Index solveCalls() const;

  Index numParams() const;
  Real  value(const Vector<Real>& prm);
  void  grad(const Vector<Real>& prm, Vector<Real>& out);
  Real  valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);

private:
  void checkDims() const;
  void solveFwd(const Vector<Real>& prm, TimeTrajectory& tr);
  void gradAt(const TimeTrajectory& tr,
              const Vector<Real>&   prm,
              Vector<Real>&         out);

  void assemble(problem::TimeContext    ctx,
                problem::VariableBlock  wrt,
                linalg::MatrixOperator& out);

  void        notify(const char* phase, Index step, Index total_steps);
  static void checkSize(const Vector<Real>& value,
                        Index               expected,
                        const char*         message);

private:
  TimeStateSolver&              state_solver_;
  const problem::TimeResidual&  eq_;
  linalg::MatrixOperator&       next_state_jac_;
  linalg::MatrixOperator&       prev_state_jac_;
  linalg::LinearSolver&         adj_solver_;
  const problem::TimeObjective& obj_;
  problem::TimeDims       dims_;
  ProgressCallback              callback_;
  InitialStateParamJacT         init_param_jac_t_;
  Real                          assembly_seconds_{0.0};
  Real                          solve_seconds_{0.0};
  Index                         assembly_calls_{0};
  Index                         solve_calls_{0};
};

} // namespace solve
} // namespace femx
