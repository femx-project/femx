#pragma once

#include <femx/common/Types.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace linalg
{
class AssemblyMatrix;
class LinearSolver;
} // namespace linalg

namespace state
{
class TimeIntegrator;
class TimeTrajectory;
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

  virtual void apply(const Vector<Real>& prm,
                     const Vector<Real>& state_grad,
                     Vector<Real>&       out) = 0;
};

class TimeReducedFunctional final
{
public:
  TimeReducedFunctional(state::TimeIntegrator&        integrator,
                        const state::TimeResidual&    problem,
                        state::TimeLinearization&     lin,
                        linalg::AssemblyMatrix&       J_next,
                        linalg::AssemblyMatrix&       J_hist,
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
  Real  value(const Vector<Real>& prm);
  void  grad(const Vector<Real>& prm, Vector<Real>& out);
  Real  valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);

private:
  void checkDims() const;
  void solveFwd(const Vector<Real>& prm, state::TimeTrajectory& tr);
  void gradAt(const state::TimeTrajectory& tr,
              const Vector<Real>&          prm,
              Vector<Real>&                out);

  void assemble(state::TimeContext      ctx,
                state::VariableBlock    wrt,
                linalg::AssemblyMatrix& out);

  void        notify(const char* phase, Index step, Index total_steps);
  static void checkSize(const Vector<Real>& value, Index exp);

private:
  state::TimeIntegrator&        integrator_;
  const state::TimeResidual&    problem_;
  state::TimeLinearization&     lin_;
  linalg::AssemblyMatrix&       J_next_;
  linalg::AssemblyMatrix&       J_hist_;
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
