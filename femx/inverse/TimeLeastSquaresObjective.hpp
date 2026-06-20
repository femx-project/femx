#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
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
                           const eq::TimeStateTrajectory& tr,
                           const Vector<Real>&            prm,
                           Vector<Real>&                  out) const;

  void obsResidual(Index                          data_row,
                   const TimeInterpolation&       interp,
                   const eq::TimeStateTrajectory& tr,
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

} // namespace inverse
} // namespace femx
