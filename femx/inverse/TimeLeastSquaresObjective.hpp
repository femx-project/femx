#pragma once

#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Objective 0.5 * sum_l weight_l ||H_l(u_l,m) - data_l||^2.
 *
 * TimeLeastSquaresObjective compares operator predictions against observation
 * data and accumulates gradients through the observation Jacobians.
 */
class TimeLeastSquaresObjective final : public TimeObjective
{
public:
  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Real                           wt = 1.0);

  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Vector<Real>                   wts);

  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Vector<Real>                   wts,
                            Real                           dt);

  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Vector<Real>                   wts,
                            Real                           dt,
                            Real                           time_offset);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const state::TimeTrajectory& tr,
             const Vector<Real>&          prm) const override;

  void stateGrad(Index                        level,
                 const state::TimeTrajectory& tr,
                 const Vector<Real>&          prm,
                 Vector<Real>&                out) const override;

  void paramGrad(const state::TimeTrajectory& tr,
                 const Vector<Real>&          prm,
                 Vector<Real>&                out) const override;

private:
  Index               numLevels() const;
  void                checkInputs() const;
  void                checkLevel(Index level) const;
  LinearInterpolation interpolation(Index row) const;
  Real                observationWeight(const LinearInterpolation& interp) const;

  void observeInterpolated(Index                        data_row,
                           const LinearInterpolation&   interp,
                           const state::TimeTrajectory& tr,
                           const Vector<Real>&          prm,
                           Vector<Real>&                out) const;

  void obsResidual(Index                        data_row,
                   const LinearInterpolation&   interp,
                   const state::TimeTrajectory& tr,
                   const Vector<Real>&          prm,
                   Vector<Real>&                out) const;

  static void checkSize(const Vector<Real>& value, Index exp);
  static void scale(Vector<Real>& out, Real factor);

private:
  const TimeObservationOperator& obs_;
  TimeObservationData            data_;
  Vector<Real>                   wts_;
  Real                           dt_{1.0};
  Real                           time_offset_{0.0};
};

} // namespace inverse
} // namespace femx
