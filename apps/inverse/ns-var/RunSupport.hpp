#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Config.hpp"
#include "DirichletControl.hpp"
#include <femx/bc/VelocityProfile.hpp>
#include <femx/common/Types.hpp>
#include <femx/eq/TimeMatrixLinearStateSolver.hpp>
#include <femx/eq/TimeStateSolver.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx::navier_var
{

struct VizOptions
{
  std::string basename = "ns-var";
};

struct InverseParameterLayout
{
  Index initial_velocity_offset = 0;
  Index initial_velocity_size   = 0;
  Index control_offset          = 0;
  Index control_size            = 0;
  Index total_size              = 0;

  bool hasInitialVelocity() const
  {
    return initial_velocity_size > 0;
  }
};

class InitialVelocityStateSolver final : public eq::TimeStateSolver
{
public:
  InitialVelocityStateSolver(eq::TimeMatrixLinearStateSolver& solver,
                             Vector<Index>                    velocity_dofs,
                             InverseParameterLayout           layout);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  void solve(const Vector<Real>&  prm,
             eq::TimeStateTrajectory& tr) override;

private:
  eq::TimeMatrixLinearStateSolver& solver_;
  Vector<Index>                    velocity_dofs_;
  InverseParameterLayout           layout_;
  Vector<Real>                     initial_state_;
};

class ParameterSliceTimeObjective final
  : public inverse::TimeObjectiveFunctional
{
public:
  ParameterSliceTimeObjective(
      const inverse::TimeObjectiveFunctional& base,
      Index                                   total_params,
      Index                                   offset);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const eq::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override;

  void stateGrad(Index                          level,
                 const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

  void paramGrad(const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

private:
  Vector<Real> slice(const Vector<Real>& prm) const;

private:
  const inverse::TimeObjectiveFunctional& base_;
  Index                                   total_params_{0};
  Index                                   offset_{0};
};

class InitialVelocityRegularization final
  : public inverse::TimeObjectiveFunctional
{
public:
  InitialVelocityRegularization(Index                         num_steps,
                                Index                         num_states,
                                InverseParameterLayout        layout,
                                Real                          beta);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const eq::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override;

  void stateGrad(Index                          level,
                 const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

  void paramGrad(const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

private:
  Index                  num_steps_{0};
  Index                  num_states_{0};
  InverseParameterLayout layout_;
  Real                   beta_{0.0};
};

std::unique_ptr<FiniteElement> makeElement(const Mesh& mesh);

MixedFESpace makeSpace(Mesh& mesh, FiniteElement& elem);

Point3 selectorCenter(const Mesh& mesh, const BoundarySelector& selector);

Vector<Index> gaugeDofs(const MixedFESpace&     space,
                        const BoundarySelector& selector);

DirichletControl makeVelocityControl(
    const MixedFESpace&     space,
    const BoundarySelector& selector);

Vector<Index> initialVelocityDofs(const MixedFESpace& space);

InverseParameterLayout inverseParameterLayout(
    const MixedFESpace&           space,
    const DirichletControl&       control,
    const InitialVelocityParams&  initial_velocity,
    Index                         steps);

Vector<Index> fixedDofs(const MixedFESpace&         space,
                        const Params&               prm,
                        const DirichletControl& control);

std::unique_ptr<inverse::TimeObservationOperator> makeObs(
    const MixedFESpace&     space,
    const ObservationParams& prm,
    Index                   steps,
    Index                   num_states,
    Index                   num_prm);

void setObsLayout(inverse::TimeObservationData& data,
                  const MixedFESpace&           space,
                  const ObservationParams&      prm);

std::unique_ptr<inverse::TimeObservationOperator> makeObsFromData(
    const MixedFESpace&                 space,
    const inverse::TimeObservationData& data,
    Index                               steps,
    Index                               num_states,
    Index                               num_prm);

Vector<Real> misfitW(Index num_steps,
                     Real  weight,
                     bool  include_initial = false);

bc::AxialVelocityProfile targetProfile(const TargetParams& target);

Real peakBaseSpeed(const TargetParams& target);

Real maxPulseSpeed(const TargetParams& target);

Real trueValue(const MixedFESpace&         space,
               const DirichletControl& control,
               const TargetParams&         target,
               Index                       step,
               Index                       i,
               Real                        dt);

Vector<Real> makeTrueParams(const MixedFESpace&         space,
                            const DirichletControl& control,
                            const TargetParams&         target,
                            Index                       steps,
                            Real                        dt);

Vector<Real> initialControlParams(const MixedFESpace&     space,
                                  const DirichletControl& control,
                                  const BCsParams&        bc,
                                  Index                   steps,
                                  Real                    dt);

Vector<Real> initialInverseParams(const MixedFESpace&           space,
                                  const DirichletControl&       control,
                                  const Params&                 prm,
                                  const InverseParameterLayout& layout,
                                  Index                         steps,
                                  Real                          dt);

void initialStateFromParams(const Vector<Index>&              velocity_dofs,
                            const InverseParameterLayout&     layout,
                            Index                             num_states,
                            const Vector<Real>&               prm,
                            Vector<Real>&                     out);

void applyInitialVelocityParamJacT(
    const Vector<Index>&              velocity_dofs,
    const InverseParameterLayout&     layout,
    const Vector<Real>&               state_grad,
    Vector<Real>&                     out);

void controlBounds(const MixedFESpace&         space,
                   const DirichletControl& control,
                   const TargetParams&         target,
                   const BoundsParams&         bounds,
                   Index                       steps,
                   Vector<Real>&               lower,
                   Vector<Real>&               upper);

void controlBounds(const MixedFESpace&     space,
                   const DirichletControl& control,
                   const BoundsParams&     bounds,
                   Index                   steps,
                   Vector<Real>&           lower,
                   Vector<Real>&           upper);

void inverseBounds(const MixedFESpace&           space,
                   const DirichletControl&       control,
                   const Params&                 prm,
                   const InverseParameterLayout& layout,
                   Index                         steps,
                   Vector<Real>&                 lower,
                   Vector<Real>&                 upper);

Real blockRmse(const DirichletControl& control,
               const Vector<Real>&         prm,
               const Vector<Real>&         target,
               Index                       step);

Index centerControlIndex(const MixedFESpace&         space,
                         const DirichletControl& control,
                         const TargetParams&         target);

void writeViz(const Mesh&                    mesh,
              const MixedFESpace&            space,
              const DirichletControl&    control,
              const eq::TimeStateTrajectory& target_tr,
              const eq::TimeStateTrajectory& opt_tr,
              const Vector<Real>&            true_prm,
              const Vector<Real>&            opt_prm,
              Real                           dt,
              const VizOptions&              opts);

void writeForwardViz(const Mesh&                    mesh,
                     const MixedFESpace&            space,
                     const eq::TimeStateTrajectory& tr,
                     Real                           dt,
                     const VizOptions&              opts);

} // namespace femx::navier_var
