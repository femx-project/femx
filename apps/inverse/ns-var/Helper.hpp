#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "IO.hpp"
#include <NavierHelper.hpp>
#include <NavierKernel.hpp>
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/assembly/TimeFEMResidual.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/SumTimeObjective.hpp>
#include <femx/problem/TimeLeastSquaresObjective.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeObservationData.hpp>
#include <femx/problem/TimeObservationOperator.hpp>
#include <femx/problem/TimeRegularization.hpp>
#include <femx/solve/TimeLinearStateSolver.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx::navier_var_new
{

using navier::gaugeDofs;
using navier::makeElement;
using navier::makeNavierKernel;
using navier::makeSpace;
using navier::makeVelocityControl;
using navier::NavierKernel;

struct InverseParameterLayout
{
  Index init_vel_offset = 0;
  Index init_vel_size   = 0;
  Index ctr_offset      = 0;
  Index ctr_levels      = 0;
  Index ctr_size        = 0;
  Index total_size      = 0;

  bool hasInitialVelocity() const
  {
    return init_vel_size > 0;
  }
};

struct FixedDofValues
{
  Vector<Index> dofs;
  Vector<Real>  values;
};

class InitialVelocityStateSolver final : public solve::TimeStateSolver
{
public:
  InitialVelocityStateSolver(
      solve::TimeLinearStateSolver& solver,
      Vector<Index>                 velocity_dofs,
      InverseParameterLayout        layout,
      Vector<Real>                  x0 = {});

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  void solve(
      const Vector<Real>&    prm,
      solve::TimeTrajectory& tr) override;

private:
  solve::TimeLinearStateSolver& solver_;
  Vector<Index>                 velocity_dofs_;
  InverseParameterLayout        layout_;
  Vector<Real>                  x0_;
  Vector<Real>                  initial_state_;
};

class ParameterSliceTimeObjective final : public problem::TimeObjective
{
public:
  ParameterSliceTimeObjective(
      const problem::TimeObjective& base,
      Index                         total_params,
      Index                         offset);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm) const override;

  void stateGrad(
      Index                        level,
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

  void paramGrad(
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

private:
  Vector<Real> slice(const Vector<Real>& prm) const;

private:
  const problem::TimeObjective& base_;
  Index                         total_params_{0};
  Index                         offset_{0};
};

class InitialVelocityRegularization final : public problem::TimeObjective
{
public:
  InitialVelocityRegularization(
      Index                  num_steps,
      Index                  num_states,
      InverseParameterLayout layout,
      Real                   beta);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm) const override;

  void stateGrad(
      Index                        level,
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

  void paramGrad(
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

private:
  Index                  num_steps_{0};
  Index                  num_states_{0};
  InverseParameterLayout layout_;
  Real                   beta_{0.0};
};

class ControlSpatialRegularization final : public problem::TimeObjective
{
public:
  using Edge = std::pair<Index, Index>;

  ControlSpatialRegularization(
      Index             num_steps,
      Index             num_states,
      Index             num_params,
      Index             ctr_offset,
      Index             ctr_levels,
      Index             ctr_dofs,
      std::vector<Edge> edges,
      Real              beta);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm) const override;

  void stateGrad(
      Index                        level,
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

  void paramGrad(
      const solve::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

private:
  Index paramIndex(Index step, Index ctr_index) const;

private:
  Index             num_steps_{0};
  Index             num_states_{0};
  Index             num_params_{0};
  Index             ctr_offset_{0};
  Index             ctr_levels_{0};
  Index             ctr_dofs_{0};
  std::vector<Edge> edges_;
  Real              beta_{0.0};
};

std::unique_ptr<problem::TimeObservationOperator> makeObs(
    const MixedFESpace&      space,
    const ObservationParams& prm,
    Index                    steps,
    Index                    num_states,
    Index                    num_prm);

void setObsLayout(
    problem::TimeObservationData& data,
    const MixedFESpace&           space,
    const ObservationParams&      prm);

Vector<Real> controlParams(
    const InverseParameterLayout& layout,
    const Vector<Real>&           prm);

Vector<Real> optimizerScale(
    const InverseParameterLayout& layout,
    const OptimizerParams::Scale& scale);

void initializeOptimizationGuess(
    const MixedFESpace&           space,
    const DirichletControl&       control,
    const Params&                 prm,
    const InverseParameterLayout& layout,
    const Vector<Index>&          velocity_dofs,
    solve::TimeLinearStateSolver& state_solver,
    const Vector<Real>&           ctr_times,
    Vector<Real>&                 prm_init,
    Vector<Real>*                 x0 = nullptr);

void applyInitialVelocityParamJacT(
    const Vector<Index>&          velocity_dofs,
    const InverseParameterLayout& layout,
    const Vector<Real>&           state_grad,
    Vector<Real>&                 out);

void inverseBounds(
    const MixedFESpace&           space,
    const DirichletControl&       control,
    const Params&                 prm,
    const InverseParameterLayout& layout,
    Index                         steps,
    Vector<Real>&                 lower,
    Vector<Real>&                 upper);

void checkInverseRunParams(const Params& prm);

struct AppNsVar
{
  explicit AppNsVar(const Params& prm);

  AppNsVar(const AppNsVar&)            = delete;
  AppNsVar& operator=(const AppNsVar&) = delete;
  AppNsVar(AppNsVar&&)                 = delete;
  AppNsVar& operator=(AppNsVar&&)      = delete;

  Index steps = 0;
  Real  dt    = 0.0;

  Mesh                                   mesh;
  std::unique_ptr<FiniteElement>         elem;
  MixedFESpace                           space;
  FluidParams                            fluid;
  GaussQuadrature                        quad;
  NavierKernel                           ns;
  assembly::TimeFEMResidual              fem;
  BCsParams                              bc;
  DirichletControl                       ctr;
  Vector<Index>                          init_vdofs;
  problem::TimeObservationData           obs_data;
  Vector<Real>                           ctr_times;
  Vector<LinearInterpolation>            ctr_time_stencils;
  InverseParameterLayout                 layout;
  FixedDofValues                         fixed;
  assembly::TimeDirichletControlResidual eq;
  Vector<Real>                           x0;
  CsrPattern                             pattern;
  Vector<Real>                           prm0;
};

struct Objective
{
  Objective(const Params& prm, const AppNsVar& problem);

  Objective(const Objective&)            = delete;
  Objective& operator=(const Objective&) = delete;
  Objective(Objective&&)                 = delete;
  Objective& operator=(Objective&&)      = delete;

  problem::TimeObservationData       data;
  fem::TimePointInterpolator         op;
  problem::TimeLeastSquaresObjective err;
  problem::TimeRegularization        ctr_reg;
  ParameterSliceTimeObjective        reg;
  InitialVelocityRegularization      init_reg;
  ControlSpatialRegularization       space_reg;
  problem::SumTimeObjective          obj;
};

} // namespace femx::navier_var_new
