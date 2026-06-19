#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/inverse/TimeLeastSquaresObjective.hpp>

using namespace femx::eq;

namespace femx
{
namespace inverse
{

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Real                           weight)
  : obs_(obs),
    data_(std::move(data))
{
  if (weight < 0.0)
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective received negative weight");
  }
  weights_.resize(numLevels());
  for (Index level = 0; level < weights_.size(); ++level)
  {
    weights_[level] = weight;
  }
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Vector<Real>                   weights)
  : obs_(obs),
    data_(std::move(data)),
    weights_(std::move(weights))
{
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Vector<Real>                   weights,
    Real                           dt)
  : obs_(obs),
    data_(std::move(data)),
    weights_(std::move(weights)),
    dt_(dt)
{
  checkInputs();
}

Index TimeLeastSquaresObjective::numSteps() const
{
  return obs_.numSteps();
}

Index TimeLeastSquaresObjective::numStates() const
{
  return obs_.numStates();
}

Index TimeLeastSquaresObjective::numParams() const
{
  return obs_.numParams();
}

Real TimeLeastSquaresObjective::value(const TimeStateTrajectory& tr,
                                      const Vector<Real>&        prm) const
{
  Real         value_out = 0.0;
  Vector<Real> res;
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    const TimeInterpolation interp = interpolation(row);
    obsResidual(row, interp, tr, prm, res);
    const Real weight = observationWeight(interp);
    for (Index i = 0; i < res.size(); ++i)
    {
      value_out += 0.5 * weight * res[i] * res[i];
    }
  }
  return value_out;
}

void TimeLeastSquaresObjective::stateGrad(
    Index                      level,
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  checkLevel(level);
  resize(out, numStates());

  Vector<Real> weighted_res;
  Vector<Real> level_grad;
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    const TimeInterpolation interp = interpolation(row);
    Real                    factor = 0.0;
    if (interp.lower == level)
    {
      factor += 1.0 - interp.upper_weight;
    }
    if (interp.upper != interp.lower && interp.upper == level)
    {
      factor += interp.upper_weight;
    }
    if (factor == 0.0)
    {
      continue;
    }

    const Real weight = observationWeight(interp);
    if (weight == 0.0)
    {
      continue;
    }

    obsResidual(row, interp, tr, prm, weighted_res);
    scale(weighted_res, factor * weight);

    obs_.applyStateJacT(level, tr[level], prm, weighted_res, level_grad);
    if (level_grad.size() != numStates())
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective state gradient size mismatch");
    }
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += level_grad[i];
    }
  }
}

void TimeLeastSquaresObjective::paramGrad(
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  resize(out, numParams());

  Vector<Real> weighted_res;
  Vector<Real> level_grad;
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    const TimeInterpolation interp = interpolation(row);
    const Real              weight = observationWeight(interp);
    if (weight == 0.0)
    {
      continue;
    }

    obsResidual(row, interp, tr, prm, weighted_res);
    scale(weighted_res, weight);

    obs_.applyParamJacT(
        interp.lower, tr[interp.lower], prm, weighted_res, level_grad);
    if (level_grad.size() != numParams())
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective parameter gradient size mismatch");
    }
    scale(level_grad, 1.0 - interp.upper_weight);
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += level_grad[i];
    }

    if (interp.upper == interp.lower || interp.upper_weight == 0.0)
    {
      continue;
    }
    obs_.applyParamJacT(
        interp.upper, tr[interp.upper], prm, weighted_res, level_grad);
    if (level_grad.size() != numParams())
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective parameter gradient size mismatch");
    }
    scale(level_grad, interp.upper_weight);
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += level_grad[i];
    }
  }
}

Index TimeLeastSquaresObjective::numLevels() const
{
  return numSteps() + 1;
}

void TimeLeastSquaresObjective::checkInputs() const
{
  if (data_.numObservations() != obs_.numObservations()
      || weights_.size() != numLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective received inconsistent time levels");
  }
  if (!data_.hasTimeLevels() && !data_.hasTimeValues()
      && data_.numLevels() != numLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective received inconsistent time levels");
  }
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    (void) interpolation(row);
  }

  for (Index level = 0; level < numLevels(); ++level)
  {
    if (weights_[level] < 0.0)
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective received negative weight");
    }
  }
}

void TimeLeastSquaresObjective::checkLevel(Index level) const
{
  if (level < 0 || level >= numLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective time level is out of range");
  }
}

TimeLeastSquaresObjective::TimeInterpolation
TimeLeastSquaresObjective::interpolation(Index row) const
{
  if (row < 0 || row >= data_.numLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective observation row is out of range");
  }
  if (data_.hasTimeValues())
  {
    if (dt_ <= 0.0 || !std::isfinite(dt_))
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective requires positive dt for time interpolation");
    }

    const Real scaled = data_.timeValue(row) / dt_;
    const Real tol =
        std::max<Real>(1.0e-10,
                       1.0e-8 * std::max<Real>(1.0, std::abs(scaled)));
    if (scaled < -tol || scaled > static_cast<Real>(numSteps()) + tol)
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective observation time is out of range");
    }

    const Real clamped =
        std::min<Real>(std::max<Real>(scaled, 0.0),
                       static_cast<Real>(numSteps()));
    const Index nearest = static_cast<Index>(std::llround(clamped));
    if (std::abs(clamped - static_cast<Real>(nearest)) <= tol)
    {
      return {nearest, nearest, 0.0};
    }

    const Index lower = static_cast<Index>(std::floor(clamped));
    const Index upper = lower + 1;
    if (lower < 0 || upper > numSteps())
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective observation time is out of range");
    }
    return {lower, upper, clamped - static_cast<Real>(lower)};
  }

  const Index level = data_.timeLevel(row);
  if (level < 0 || level >= numLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective observation time level is out of range");
  }
  return {level, level, 0.0};
}

Real TimeLeastSquaresObjective::observationWeight(
    const TimeInterpolation& interp) const
{
  return (1.0 - interp.upper_weight) * weights_[interp.lower]
         + interp.upper_weight * weights_[interp.upper];
}

void TimeLeastSquaresObjective::observeInterpolated(
    Index                      data_row,
    const TimeInterpolation&   interp,
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  (void) data_row;
  obs_.observe(interp.lower, tr[interp.lower], prm, out);
  if (out.size() != obs_.numObservations())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective observation size mismatch");
  }

  if (interp.upper == interp.lower || interp.upper_weight == 0.0)
  {
    return;
  }

  Vector<Real> upper;
  obs_.observe(interp.upper, tr[interp.upper], prm, upper);
  if (upper.size() != obs_.numObservations())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective observation size mismatch");
  }

  const Real lower_weight = 1.0 - interp.upper_weight;
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = lower_weight * out[i] + interp.upper_weight * upper[i];
  }
}

void TimeLeastSquaresObjective::obsResidual(
    Index                      data_row,
    const TimeInterpolation&   interp,
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  if (data_row < 0 || data_row >= data_.numLevels()
      || tr.numLevels() != numLevels() || tr.numStates() != numStates())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective trajectory size mismatch");
  }

  observeInterpolated(data_row, interp, tr, prm, out);

  const Vector<Real> data = data_[data_row];
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] -= data[i];
  }
}

void TimeLeastSquaresObjective::resize(Vector<Real>& out,
                                       Index         size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

void TimeLeastSquaresObjective::scale(Vector<Real>& out,
                                      Real          factor)
{
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] *= factor;
  }
}

} // namespace inverse
} // namespace femx
