#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/operator/LinearSolver.hpp>
#include <femx/linalg/operator/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeResidual.hpp>
#include <femx/state/TimeIntegrator.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace state
{

class TimeReducedProgressMonitor
{
public:
  virtual ~TimeReducedProgressMonitor() = default;

  virtual void progress(const char* phase,
                        Index       step,
                        Index       total_steps) = 0;
};

class InitialStateGradientMap
{
public:
  virtual ~InitialStateGradientMap() = default;

  virtual void apply(const Vector<Real>& prm,
                     const Vector<Real>& state_grad,
                     Vector<Real>&       out) = 0;
};

class TimeReducedFunctional final
{
public:
  TimeReducedFunctional(TimeIntegrator&               integrator,
                        const problem::TimeResidual&  problem,
                        problem::TimeLinearization&   lin,
                        linalg::MatrixOperator&       J_next,
                        linalg::MatrixOperator&       J_hist,
                        linalg::LinearSolver&         adj_solver,
                        const problem::TimeObjective& obj);

  void setMonitor(TimeReducedProgressMonitor* monitor);
  void clearMonitor();

  void setInitialStateParamJacT(InitialStateGradientMap* map);
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
  static void checkSize(const Vector<Real>& value, Index exp);

private:
  TimeIntegrator&               integrator_;
  const problem::TimeResidual&  problem_;
  problem::TimeLinearization&   lin_;
  linalg::MatrixOperator&       J_next_;
  linalg::MatrixOperator&       J_hist_;
  linalg::LinearSolver&         adj_solver_;
  const problem::TimeObjective& obj_;
  problem::TimeDims             dims_;
  TimeReducedProgressMonitor*   progress_monitor_{nullptr};
  InitialStateGradientMap*      init_grad_map_{nullptr};
  Real                          assm_sec_{0.0};
  Real                          solve_sec_{0.0};
  Index                         assm_calls_{0};
  Index                         solve_calls_{0};
};

} // namespace state
} // namespace femx
