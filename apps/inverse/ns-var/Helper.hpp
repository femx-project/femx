#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

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
#include <femx/linalg/VectorView.hpp>
#include <femx/problem/SumTimeObjective.hpp>
#include <femx/problem/TimeLeastSquaresObjective.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeObservationData.hpp>
#include <femx/problem/TimeObservationOperator.hpp>
#include <femx/problem/TimeRegularization.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>
#include <femx/state/TimeStateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

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
  Index niv             = 0;
  Index coff            = 0;
  Index nctr            = 0;
  Index csz             = 0;
  Index ntot            = 0;

  bool hasInitialVelocity() const
  {
    return niv > 0;
  }

  void checkPrm(const Vector<Real>& prm) const
  {
    if (prm.size() != ntot)
    {
      throw std::runtime_error("InverseParameterLayout parameter size mismatch");
    }
  }

  VectorView<Real> initVel(Vector<Real>& prm) const
  {
    checkPrm(prm);
    return VectorView<Real>(prm.data() + init_vel_offset, niv);
  }

  VectorView<const Real> initVel(const Vector<Real>& prm) const
  {
    checkPrm(prm);
    return VectorView<const Real>(prm.data() + init_vel_offset, niv);
  }

  VectorView<Real> ctr(Vector<Real>& prm) const
  {
    checkPrm(prm);
    return VectorView<Real>(prm.data() + coff, csz);
  }

  VectorView<const Real> ctr(const Vector<Real>& prm) const
  {
    checkPrm(prm);
    return VectorView<const Real>(prm.data() + coff, csz);
  }
};

struct FixedDofValues
{
  Vector<Index> dofs;
  Vector<Real>  vals;
};

class InitialVelocityStateSolver final : public state::TimeStateSolver
{
public:
  InitialVelocityStateSolver(
      state::TimeLinearStateSolver& solver,
      Vector<Index>                 vdofs,
      InverseParameterLayout        lyt,
      Vector<Real>                  x0 = {});

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  void solve(
      const Vector<Real>&    prm,
      state::TimeTrajectory& tr) override;

private:
  state::TimeLinearStateSolver& solver_;
  Vector<Index>                 vdofs_;
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
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm) const override;

  void stateGrad(
      Index                        level,
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

  void paramGrad(
      const state::TimeTrajectory& tr,
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
      Index                  nt,
      Index                  nst,
      InverseParameterLayout lyt,
      Real                   beta);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm) const override;

  void stateGrad(
      Index                        level,
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

  void paramGrad(
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

private:
  Index                  nt_{0};
  Index                  nst_{0};
  InverseParameterLayout layout_;
  Real                   beta_{0.0};
};

class ControlSpatialRegularization final : public problem::TimeObjective
{
public:
  using Edge = std::pair<Index, Index>;

  ControlSpatialRegularization(
      Index        nt,
      Index        nst,
      Index        nprm,
      Index        coff,
      Index        nctr,
      Index        ncdof,
      Vector<Edge> edges,
      Real         beta);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm) const override;

  void stateGrad(
      Index                        level,
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

  void paramGrad(
      const state::TimeTrajectory& tr,
      const Vector<Real>&          prm,
      Vector<Real>&                out) const override;

private:
  Index paramIndex(Index step, Index ctr_index) const;

private:
  Index        nt_{0};
  Index        nst_{0};
  Index        nprm_{0};
  Index        ctr_offset_{0};
  Index        ctr_levels_{0};
  Index        ctr_dofs_{0};
  Vector<Edge> edges_;
  Real         beta_{0.0};
};

std::unique_ptr<problem::TimeObservationOperator> makeObs(
    const MixedFESpace&      space,
    const ObservationParams& prm,
    Index                    steps,
    Index                    nst,
    Index                    nprm);

void setObsLayout(
    problem::TimeObservationData& data,
    const MixedFESpace&           space,
    const ObservationParams&      prm);

Vector<Real> controlParams(
    const InverseParameterLayout& lyt,
    const Vector<Real>&           prm);

Vector<Real> optimizerScale(
    const InverseParameterLayout& lyt,
    const OptimizerParams::Scale& scale);

void initializeOptGuess(
    const MixedFESpace&           space,
    const DirichletControl&       ctr,
    const Params&                 prm,
    const InverseParameterLayout& lyt,
    const Vector<Index>&          vdofs,
    state::TimeLinearStateSolver& state_solver,
    const Vector<Real>&           ctr_times,
    Vector<Real>&                 prm_init,
    Vector<Real>*                 x0 = nullptr);

void applyInitialVelocityParamJacT(
    const Vector<Index>&          vdofs,
    const InverseParameterLayout& lyt,
    const Vector<Real>&           state_grad,
    Vector<Real>&                 out);

void inverseBounds(
    const MixedFESpace&           space,
    const DirichletControl&       ctr,
    const Params&                 prm,
    const InverseParameterLayout& lyt,
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
  InverseParameterLayout                 lyt;
  FixedDofValues                         fixed;
  assembly::TimeDirichletControlResidual problem;
  Vector<Real>                           x0;
  CsrPattern                             pettern;
  Vector<Real>                           prm0;
};

struct Objective
{
  Objective(const Params& prm, const AppNsVar& app);

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
