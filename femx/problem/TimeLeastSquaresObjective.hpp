#pragma once

#include <femx/core/Types.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/problem/TimeObjectiveFunctional.hpp>
#include <femx/problem/TimeObservationData.hpp>
#include <femx/problem/TimeObservationOperator.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Objective 0.5 * sum_l weight_l ||H_l(u_l,m) - data_l||^2. */
class TimeLeastSquaresObjective final : public TimeObjectiveFunctional
{
public:
  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Real                           weight = 1.0);

  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Vector<Real>                   weights);

  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Vector<Real>                   weights,
                            Real                           dt);

  TimeLeastSquaresObjective(const TimeObservationOperator& obs,
                            TimeObservationData            data,
                            Vector<Real>                   weights,
                            Real                           dt,
                            Real                           time_offset);

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Real value(const solve::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override;

  void stateGrad(Index                          level,
                 const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

  void paramGrad(const solve::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

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

  void observeInterpolated(Index                          data_row,
                           const TimeInterpolation&       interp,
                           const solve::TimeStateTrajectory& tr,
                           const Vector<Real>&            prm,
                           Vector<Real>&                  out) const;

  void obsResidual(Index                          data_row,
                   const TimeInterpolation&       interp,
                   const solve::TimeStateTrajectory& tr,
                   const Vector<Real>&            prm,
                   Vector<Real>&                  out) const;

  static void resize(Vector<Real>& out,
                     Index         size);

  static void scale(Vector<Real>& out,
                    Real          factor);

private:
  const TimeObservationOperator& obs_;
  TimeObservationData            data_;
  Vector<Real>                   weights_;
  Real                           dt_{1.0};
  Real                           time_offset_{0.0};
};

} // namespace problem
} // namespace femx
