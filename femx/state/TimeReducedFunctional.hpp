#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/TimeStateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace state
{

class TimeReducedFunctional final
{
public:
  using ProgressCallback = std::function<void(const char* phase, Index step, Index total_steps)>;

  using InitialStateParamJacT =
      std::function<void(const Vector<Real>& prm,
                         const Vector<Real>& st_grad,
                         Vector<Real>&       out)>;

  TimeReducedFunctional(TimeStateSolver&              state_solver,
                        const problem::TimeResidual&  problem,
                        problem::TimeLinearization&   lin,
                        linalg::MatrixOperator&       nxt_jac,
                        linalg::MatrixOperator&       hist_jac,
                        linalg::LinearSolver&         adj_solver,
                        const problem::TimeObjective& obj);

  void setProgress(ProgressCallback cb);
  void clearProgress();

  void setInitialStateParamJacT(InitialStateParamJacT cb);
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
                        Index               exp);

private:
  TimeStateSolver&              state_solver_;
  const problem::TimeResidual&  problem_;
  problem::TimeLinearization&   lin_;
  linalg::MatrixOperator&       nxt_jac_;
  linalg::MatrixOperator&       hist_jac_;
  linalg::LinearSolver&         adj_solver_;
  const problem::TimeObjective& obj_;
  problem::TimeDims             dims_;
  ProgressCallback              callback_;
  InitialStateParamJacT         init_param_jac_t_;
  Real                          assembly_seconds_{0.0};
  Real                          solve_seconds_{0.0};
  Index                         assembly_calls_{0};
  Index                         solve_calls_{0};
};

} // namespace state
} // namespace femx
