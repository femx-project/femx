#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/inverse/TimeLeastSquaresObjective.hpp>
using namespace femx::state;

namespace femx
{
namespace inverse
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
    throw std::runtime_error(
        "TimeLeastSquaresObjective received negative weight");
  }
  wts_.resize(numTimeLevels());
  for (Index level = 0; level < wts_.size(); ++level)
  {
    wts_[level] = wt;
  }
  setUniformObservationWeights();
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    HostVector                     wts)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts))
{
  setUniformObservationWeights();
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    HostVector                     wts,
    Real                           dt)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts)),
    dt_(dt)
{
  setUniformObservationWeights();
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    HostVector                     wts,
    Real                           dt,
    Real                           time_offset)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts)),
    dt_(dt),
    time_offset_(time_offset)
{
  setUniformObservationWeights();
  checkInputs();
}

TimeLeastSquaresObjective::TimeLeastSquaresObjective(
    const TimeObservationOperator& obs,
    TimeObservationData            data,
    HostVector                     wts,
    HostVector                     obs_wts,
    Real                           dt,
    Real                           time_offset)
  : obs_(obs),
    data_(std::move(data)),
    wts_(std::move(wts)),
    obs_wts_(std::move(obs_wts)),
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
                                      const HostVector&     prm) const
{
  Real       value_out = 0.0;
  HostVector res;
  for (Index row = 0; row < data_.numTimeLevels(); ++row)
  {
    const LinearInterpolation interp = interpolation(row);
    obsResidual(row, interp, tr, prm, res);
    const Real wt = observationWeight(interp);
    for (Index i = 0; i < res.size(); ++i)
    {
      value_out += 0.5 * wt * observationEntryWeight(row, i)
                   * res[i] * res[i];
    }
  }
  return value_out;
}

void TimeLeastSquaresObjective::stateGrad(Index                 level,
                                          const TimeTrajectory& tr,
                                          const HostVector&     prm,
                                          HostVector&           out) const
{
  checkLevel(level);
  resizeOrZero(out, numStates());

  HostVector weighted_res;
  HostVector level_grad;
  for (Index row = 0; row < data_.numTimeLevels(); ++row)
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
    scaleObservationResidual(row, weighted_res, factor * wt);
    obs_.applyStateJacT(level, tr[level], prm, weighted_res, level_grad);
    checkSize(level_grad, numStates());
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] += level_grad[i];
    }
  }
}

void TimeLeastSquaresObjective::paramGrad(const TimeTrajectory& tr,
                                          const HostVector&     prm,
                                          HostVector&           out) const
{
  resizeOrZero(out, numParams());

  HostVector weighted_res;
  HostVector level_grad;
  for (Index row = 0; row < data_.numTimeLevels(); ++row)
  {
    const LinearInterpolation interp = interpolation(row);
    const Real                wt     = observationWeight(interp);
    if (wt == 0.0)
    {
      continue;
    }

    obsResidual(row, interp, tr, prm, weighted_res);
    scaleObservationResidual(row, weighted_res, wt);

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

Index TimeLeastSquaresObjective::numTimeLevels() const
{
  return numSteps() + 1;
}

void TimeLeastSquaresObjective::checkInputs() const
{
  if (data_.numObservations() != obs_.numObservations()
      || wts_.size() != numTimeLevels()
      || obs_wts_.size()
             != data_.numTimeLevels() * data_.numObservations())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective received inconsistent dimensions");
  }
  if (!std::isfinite(time_offset_))
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective received invalid time offset");
  }
  if (!data_.hasTimeLevels() && !data_.hasTimeValues()
      && data_.numTimeLevels() != numTimeLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective received inconsistent time levels");
  }
  for (Index row = 0; row < data_.numTimeLevels(); ++row)
  {
    (void) interpolation(row);
  }
  for (Index level = 0; level < wts_.size(); ++level)
  {
    if (!std::isfinite(wts_[level]) || wts_[level] < 0.0)
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective received invalid weight");
    }
  }
  for (Real weight : obs_wts_)
  {
    if (!std::isfinite(weight) || weight < 0.0)
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective received invalid observation weight");
    }
  }
}

void TimeLeastSquaresObjective::checkLevel(Index level) const
{
  if (level < 0 || level >= numTimeLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective time level is out of range");
  }
}

LinearInterpolation
TimeLeastSquaresObjective::interpolation(Index row) const
{
  if (row < 0 || row >= data_.numTimeLevels())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective observation row is out of range");
  }
  if (data_.hasTimeValues())
  {
    if (dt_ <= 0.0 || !std::isfinite(dt_))
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective requires positive dt");
    }
    const Real scaled = (data_.timeValue(row) + time_offset_) / dt_;
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
    const Index nearest = static_cast<Index>(llround(clamped));
    if (std::abs(clamped - static_cast<Real>(nearest)) <= tol)
    {
      return {nearest, nearest, 0.0};
    }
    const Index lower = static_cast<Index>(floor(clamped));
    return {lower, lower + 1, clamped - static_cast<Real>(lower)};
  }

  Index level = data_.timeLevel(row);
  if (time_offset_ != 0.0)
  {
    if (dt_ <= 0.0 || !std::isfinite(dt_))
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective requires positive dt for time offset");
    }
    const Real  offset       = time_offset_ / dt_;
    const Index level_offset = static_cast<Index>(llround(offset));
    if (std::abs(offset - static_cast<Real>(level_offset)) > 1.0e-8)
    {
      throw std::runtime_error(
          "TimeLeastSquaresObjective time offset must align to a time step");
    }
    level += level_offset;
  }
  if (level < 0 || level >= numTimeLevels())
  {
    throw std::runtime_error(
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

Real TimeLeastSquaresObjective::observationEntryWeight(
    Index row,
    Index observation) const
{
  if (row < 0 || row >= data_.numTimeLevels() || observation < 0
      || observation >= data_.numObservations())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective observation weight index is out of range");
  }
  return obs_wts_[row * data_.numObservations() + observation];
}

void TimeLeastSquaresObjective::observeInterpolated(
    Index                      data_row,
    const LinearInterpolation& interp,
    const TimeTrajectory&      tr,
    const HostVector&          prm,
    HostVector&                out) const
{
  (void) data_row;
  obs_.observe(interp.lower, tr[interp.lower], prm, out);
  checkSize(out, obs_.numObservations());
  if (!interp.hasUpper())
  {
    return;
  }

  HostVector upper;
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
    const HostVector&          prm,
    HostVector&                out) const
{
  if (tr.numTimeLevels() != numTimeLevels() || tr.numStates() != numStates())
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective trajectory size mismatch");
  }
  observeInterpolated(data_row, interp, tr, prm, out);

  const HostVector data = data_[data_row];
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] -= data[i];
  }
}

void TimeLeastSquaresObjective::checkSize(const HostVector& value,
                                          Index             exp)
{
  if (value.size() != exp)
  {
    throw std::runtime_error(
        "TimeLeastSquaresObjective vector size mismatch");
  }
}

void TimeLeastSquaresObjective::scale(HostVector& out, Real factor)
{
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] *= factor;
  }
}

void TimeLeastSquaresObjective::scaleObservationResidual(
    Index       row,
    HostVector& out,
    Real        factor) const
{
  checkSize(out, data_.numObservations());
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] *= factor * observationEntryWeight(row, i);
  }
}

void TimeLeastSquaresObjective::setUniformObservationWeights()
{
  obs_wts_.resize(
      data_.numTimeLevels() * data_.numObservations());
  for (Real& weight : obs_wts_)
  {
    weight = 1.0;
  }
}

} // namespace inverse
} // namespace femx
