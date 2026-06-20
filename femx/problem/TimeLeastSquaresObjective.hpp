#pragma once

#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/TimeObjective.hpp>
#include <femx/problem/TimeObservation.hpp>
#include <femx/problem/TimeObservationData.hpp>
#include <femx/solve/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

/** @brief Objective 0.5 * sum_l weight_l ||H_l(u_l,m) - data_l||^2. */
class TimeLeastSquaresObjective final : public TimeObjective
{
public:
  TimeLeastSquaresObjective(const TimeObservation& obs,
                            TimeObservationData data,
                            Real weight = 1.0);

  TimeLeastSquaresObjective(const TimeObservation& obs,
                            TimeObservationData data,
                            Vector<Real> weights);

  TimeLeastSquaresObjective(const TimeObservation& obs,
                            TimeObservationData data,
                            Vector<Real> weights,
                            Real dt);

  TimeLeastSquaresObjective(const TimeObservation& obs,
                            TimeObservationData data,
                            Vector<Real> weights,
                            Real dt,
                            Real time_offset);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;

  Real value(const solve::TimeTrajectory& tr,
             const Vector<Real>& prm) const override;

  void stateGrad(Index level,
                 const solve::TimeTrajectory& tr,
                 const Vector<Real>& prm,
                 Vector<Real>& out) const override;

  void paramGrad(const solve::TimeTrajectory& tr,
                 const Vector<Real>& prm,
                 Vector<Real>& out) const override;

private:
  struct TimeInterpolation
  {
    Index lower        = 0;
    Index upper        = 0;
    Real  upper_weight = 0.0;
  };

  Index numLevels() const;
  void checkInputs() const;
  void checkLevel(Index level) const;
  TimeInterpolation interpolation(Index row) const;
  Real observationWeight(const TimeInterpolation& interp) const;

  void observeInterpolated(Index data_row,
                           const TimeInterpolation& interp,
                           const solve::TimeTrajectory& tr,
                           const Vector<Real>& prm,
                           Vector<Real>& out) const;

  void obsResidual(Index data_row,
                   const TimeInterpolation& interp,
                   const solve::TimeTrajectory& tr,
                   const Vector<Real>& prm,
                   Vector<Real>& out) const;

  static void resize(Vector<Real>& out, Index size);
  static void scale(Vector<Real>& out, Real factor);

private:
  const TimeObservation& obs_;
  TimeObservationData    data_;
  Vector<Real>           weights_;
  Real                   dt_{1.0};
  Real                   time_offset_{0.0};
};

} // namespace problem
} // namespace femx
