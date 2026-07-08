#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/problem/TimeLeastSquaresObjective.hpp>

using namespace std;
using namespace femx::state;

namespace femx
{
namespace problem
{

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Real                           wt)
  : obs_(obs),
    data_(std::move(data))
{
  if (wt < 0.0)
  {
    throw runtime_error(
        "TimeLeastSquaresObjective received negative weight");
  }
  wts_.resize(numLevels());
  for (Index level = 0; level < wts_.size(); ++level)
  {
    wts_[level] = wt;
  }
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Vector<Real>                   wts)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts))
{
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Vector<Real>                   wts,
    Real                           dt)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts)),
    dt_(dt)
{
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    Vector<Real>                   wts,
    Real                           dt,
    Real                           time_offset)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts)),
    dt_(dt),
    time_offset_(time_offset)
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

Real TimeLeastSquaresObjective::value(const TimeTrajectory& tr,
                                      const Vector<Real>&   prm) const
{
  Real         value_out = 0.0;
  Vector<Real> res;
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    const LinearInterpolation interp = interpolation(row);
    obsResidual(row, interp, tr, prm, res);
    const Real wt = observationWeight(interp);
    for (Index i = 0; i < res.size(); ++i)
    {
      value_out += 0.5 * wt * res[i] * res[i];
    }
  }
  return value_out;
}

void TimeLeastSquaresObjective::stateGrad(Index                 level,
                                          const TimeTrajectory& tr,
                                          const Vector<Real>&   prm,
                                          Vector<Real>&         out) const
{
  checkLevel(level);
  resizeOrZero(out, numStates());

  Vector<Real> weighted_res;
  Vector<Real> level_grad;
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    const LinearInterpolation interp = interpolation(row);
    Real                      factor = 0.0;
    interp.forEachWeight(
        [&](Index interp_level, Real wt)
        {
          if (interp_level == level)
          {
            factor += wt;
          }
        });
    if (factor == 0.0)
    {
      continue;
    }

    const Real wt = observationWeight(interp);
    if (wt == 0.0)
    {
      continue;
    }

    obsResidual(row, interp, tr, prm, weighted_res);
    scale(weighted_res, factor * wt);
    obs_.applyStateJacT(level, tr[level], prm, weighted_res, level_grad);
    checkSize(level_grad, numStates());
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += level_grad[i];
    }
  }
}

void TimeLeastSquaresObjective::paramGrad(const TimeTrajectory& tr,
                                          const Vector<Real>&   prm,
                                          Vector<Real>&         out) const
{
  resizeOrZero(out, numParams());

  Vector<Real> weighted_res;
  Vector<Real> level_grad;
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    const LinearInterpolation interp = interpolation(row);
    const Real                wt     = observationWeight(interp);
    if (wt == 0.0)
    {
      continue;
    }

    obsResidual(row, interp, tr, prm, weighted_res);
    scale(weighted_res, wt);

    interp.forEachWeight(
        [&](Index interp_level, Real interp_weight)
        {
          obs_.applyParamJacT(
              interp_level,
              tr[interp_level],
              prm,
              weighted_res,
              level_grad);
          checkSize(level_grad, numParams());
          scale(level_grad, interp_weight);
          for (Index i = 0; i < out.size(); ++i)
          {
            out[i] += level_grad[i];
          }
        });
  }
}

Index TimeLeastSquaresObjective::numLevels() const
{
  return numSteps() + 1;
}

void TimeLeastSquaresObjective::checkInputs() const
{
  if (data_.numObservations() != obs_.numObservations()
      || wts_.size() != numLevels())
  {
    throw runtime_error(
        "TimeLeastSquaresObjective received inconsistent dimensions");
  }
  if (!isfinite(time_offset_))
  {
    throw runtime_error(
        "TimeLeastSquaresObjective received invalid time offset");
  }
  if (!data_.hasTimeLevels() && !data_.hasTimeValues()
      && data_.numLevels() != numLevels())
  {
    throw runtime_error(
        "TimeLeastSquaresObjective received inconsistent time levels");
  }
  for (Index row = 0; row < data_.numLevels(); ++row)
  {
    (void) interpolation(row);
  }
  for (Index level = 0; level < wts_.size(); ++level)
  {
    if (wts_[level] < 0.0)
    {
      throw runtime_error(
          "TimeLeastSquaresObjective received negative weight");
    }
  }
}

void TimeLeastSquaresObjective::checkLevel(Index level) const
{
  if (level < 0 || level >= numLevels())
  {
    throw runtime_error(
        "TimeLeastSquaresObjective time level is out of range");
  }
}

LinearInterpolation
TimeLeastSquaresObjective::interpolation(Index row) const
{
  if (row < 0 || row >= data_.numLevels())
  {
    throw runtime_error(
        "TimeLeastSquaresObjective observation row is out of range");
  }
  if (data_.hasTimeValues())
  {
    if (dt_ <= 0.0 || !isfinite(dt_))
    {
      throw runtime_error(
          "TimeLeastSquaresObjective requires positive dt");
    }
    const Real scaled = (data_.timeValue(row) + time_offset_) / dt_;
    const Real tol =
        max<Real>(1.0e-10,
                  1.0e-8 * max<Real>(1.0, abs(scaled)));
    if (scaled < -tol || scaled > static_cast<Real>(numSteps()) + tol)
    {
      throw runtime_error(
          "TimeLeastSquaresObjective observation time is out of range");
    }

    const Real clamped =
        min<Real>(max<Real>(scaled, 0.0),
                  static_cast<Real>(numSteps()));
    const Index nearest = static_cast<Index>(llround(clamped));
    if (abs(clamped - static_cast<Real>(nearest)) <= tol)
    {
      return {nearest, nearest, 0.0};
    }
    const Index lower = static_cast<Index>(floor(clamped));
    return {lower, lower + 1, clamped - static_cast<Real>(lower)};
  }

  Index level = data_.timeLevel(row);
  if (time_offset_ != 0.0)
  {
    if (dt_ <= 0.0 || !isfinite(dt_))
    {
      throw runtime_error(
          "TimeLeastSquaresObjective requires positive dt for time offset");
    }
    const Real  offset       = time_offset_ / dt_;
    const Index level_offset = static_cast<Index>(llround(offset));
    if (abs(offset - static_cast<Real>(level_offset)) > 1.0e-8)
    {
      throw runtime_error(
          "TimeLeastSquaresObjective time offset must align to a time step");
    }
    level += level_offset;
  }
  if (level < 0 || level >= numLevels())
  {
    throw runtime_error(
        "TimeLeastSquaresObjective observation time level is out of range");
  }
  return {level, level, 0.0};
}

Real TimeLeastSquaresObjective::observationWeight(const LinearInterpolation& interp) const
{
  Real out = 0.0;
  interp.forEachWeight(
      [&](Index level, Real wt)
      {
        out += wt * wts_[level];
      });
  return out;
}

void TimeLeastSquaresObjective::observeInterpolated(
    Index                      data_row,
    const LinearInterpolation& interp,
    const TimeTrajectory&      tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  (void) data_row;
  obs_.observe(interp.lower, tr[interp.lower], prm, out);
  checkSize(out, obs_.numObservations());
  if (!interp.hasUpper())
  {
    return;
  }

  Vector<Real> upper;
  obs_.observe(interp.upper, tr[interp.upper], prm, upper);
  checkSize(upper, obs_.numObservations());
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] = interp.lowerWeight() * out[i] + interp.upperWeight() * upper[i];
  }
}

void TimeLeastSquaresObjective::obsResidual(
    Index                      data_row,
    const LinearInterpolation& interp,
    const TimeTrajectory&      tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  if (tr.numLevels() != numLevels() || tr.numStates() != numStates())
  {
    throw runtime_error(
        "TimeLeastSquaresObjective trajectory size mismatch");
  }
  observeInterpolated(data_row, interp, tr, prm, out);

  const Vector<Real> data = data_[data_row];
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] -= data[i];
  }
}

void TimeLeastSquaresObjective::checkSize(const Vector<Real>& value,
                                          Index               exp)
{
  if (value.size() != exp)
  {
    throw runtime_error(
        "TimeLeastSquaresObjective vector size mismatch");
  }
}

void TimeLeastSquaresObjective::scale(Vector<Real>& out, Real factor)
{
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] *= factor;
  }
}

} // namespace problem
} // namespace femx
