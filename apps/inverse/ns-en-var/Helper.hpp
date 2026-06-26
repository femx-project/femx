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
#include <femx/problem/TimeLeastSquaresObjective.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeObservationData.hpp>
#include <femx/problem/TimeObservationOperator.hpp>
#include <femx/state/EnsembleBasis.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>
#include <femx/state/TimeStateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx::navier_en_var
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

Index observationVectorSize(
    const problem::TimeObservationData& data);

Vector<Real> flattenObservations(
    const problem::TimeObservationData& data);

problem::TimeObservationData sampleObservationData(
    const problem::TimeObservationOperator& obs,
    const problem::TimeObservationData&     layout,
    const state::TimeTrajectory&            tr,
    const Vector<Real>&                     prm,
    Real                                    dt,
    Real                                    time_offset = 0.0);

Vector<Real> controlParams(
    const InverseParameterLayout& lyt,
    const Vector<Real>&           prm);

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

state::EnsembleBasis ensembleBasis(
    const EnsembleParams& prm,
    const Vector<Real>&   fallback_mean,
    Index                 nprm);

state::EnsembleBasis observationEnsembleBasis(
    const EnsembleParams& prm,
    Index                 nobs,
    Index                 ncoef);

void checkInverseRunParams(const Params& prm);

struct AppNsEnVar
{
  explicit AppNsEnVar(const Params& prm);

  AppNsEnVar(const AppNsEnVar&)            = delete;
  AppNsEnVar& operator=(const AppNsEnVar&) = delete;
  AppNsEnVar(AppNsEnVar&&)                 = delete;
  AppNsEnVar& operator=(AppNsEnVar&&)      = delete;

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
  Objective(const Params& prm, const AppNsEnVar& app);

  Objective(const Objective&)            = delete;
  Objective& operator=(const Objective&) = delete;
  Objective(Objective&&)                 = delete;
  Objective& operator=(Objective&&)      = delete;

  problem::TimeObservationData       data;
  fem::TimePointInterpolator         op;
  problem::TimeLeastSquaresObjective err;
};

} // namespace femx::navier_en_var
