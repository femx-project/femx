#pragma once

#include <femx/common/Types.hpp>
#include <femx/state/TimeResidual.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace linalg
{
class MatrixOperator;
class LinearSolver;
} // namespace linalg

namespace state
{
class TimeIntegrator;
} // namespace state

namespace inverse
{

class TimeObjective;

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

  virtual void apply(const HostVector& prm,
                     const HostVector& state_grad,
                     HostVector&       out) = 0;
};

class TimeReducedFunctional final
{
public:
  TimeReducedFunctional(state::TimeIntegrator&        integrator,
                        const state::TimeResidual&    problem,
                        state::TimeLinearization&     lin,
                        linalg::MatrixOperator&       J_next,
                        linalg::MatrixOperator&       J_hist,
                        linalg::LinearSolver&         adj_solver,
                        const inverse::TimeObjective& obj);

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
  Real  value(const HostVector& prm);
  void  grad(const HostVector& prm, HostVector& out);
  Real  valueGrad(const HostVector& prm, HostVector& grad_out);

private:
  void checkDims() const;
  void solveFwd(const HostVector& prm, state::TimeTrajectory& tr);
  void gradAt(const state::TimeTrajectory& tr,
              const HostVector&            prm,
              HostVector&                  out);

  void assemble(state::TimeContext      ctx,
                state::VariableBlock    wrt,
                linalg::MatrixOperator& out);

  void        notify(const char* phase, Index step, Index total_steps);
  static void checkSize(const HostVector& value, Index exp);

private:
  state::TimeIntegrator&        integrator_;
  const state::TimeResidual&    problem_;
  state::TimeLinearization&     lin_;
  linalg::MatrixOperator&       J_next_;
  linalg::MatrixOperator&       J_hist_;
  linalg::LinearSolver&         adj_solver_;
  const inverse::TimeObjective& obj_;
  state::TimeDims               dims_;
  TimeReducedProgressMonitor*   progress_monitor_{nullptr};
  InitialStateGradientMap*      init_grad_map_{nullptr};
  Real                          assm_sec_{0.0};
  Real                          solve_sec_{0.0};
  Index                         assm_calls_{0};
  Index                         solve_calls_{0};
};

} // namespace inverse
} // namespace femx
