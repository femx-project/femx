#include <cmath>
#include <stdexcept>

#include <femx/inverse/TimeRegularization.hpp>
using namespace femx::state;

namespace femx
{
namespace inverse
{

TimeRegularization::TimeRegularization(Index             num_steps,
                                       Index             num_states,
                                       Index             num_levels,
                                       Index             block_size,
                                       Real              beta_difference,
                                       Real              beta_value,
                                       const HostVector& reference)
  : num_steps_(num_steps),
    num_states_(num_states),
    num_levels_(num_levels),
    block_size_(block_size),
    beta_difference_(beta_difference),
    beta_value_(beta_value)
{
  if (num_steps_ < 0 || num_states_ < 0 || num_levels_ < 0
      || block_size_ < 0 || !std::isfinite(beta_difference_)
      || !std::isfinite(beta_value_) || beta_difference_ < 0.0
      || beta_value_ < 0.0)
  {
    throw std::runtime_error("TimeRegularization received invalid inputs");
  }
  if (reference.empty())
  {
    reference_.resize(numParams());
  }
  else if (reference.size() == numParams())
  {
    for (const Real value : reference)
    {
      if (!std::isfinite(value))
      {
        throw std::runtime_error(
            "TimeRegularization reference must be finite");
      }
    }
    reference_ = reference;
  }
  else
  {
    throw std::runtime_error("TimeRegularization reference size mismatch");
  }
}

Index TimeRegularization::numSteps() const
{
  return num_steps_;
}

Index TimeRegularization::numStates() const
{
  return num_states_;
}

Index TimeRegularization::numParams() const
{
  return num_levels_ * block_size_;
}

Real TimeRegularization::value(const TimeTrajectory& tr,
                               const HostVector&     prm) const
{
  (void) tr;
  checkParamSize(prm);

  Real value_out = 0.0;
  for (Index i = 0; i < numParams(); ++i)
  {
    const Real diff  = prm[i] - reference_[i];
    value_out       += 0.5 * beta_value_ * diff * diff;
  }
  for (Index level = 1; level < num_levels_; ++level)
  {
    for (Index c = 0; c < block_size_; ++c)
    {
      const Real diff  = centered(prm, level, c) - centered(prm, level - 1, c);
      value_out       += 0.5 * beta_difference_ * diff * diff;
    }
  }
  return value_out;
}

void TimeRegularization::stateGrad(Index                 level,
                                   const TimeTrajectory& tr,
                                   const HostVector&     prm,
                                   HostVector&           out) const
{
  (void) level;
  (void) tr;
  (void) prm;
  resizeOrZero(out, numStates());
}

void TimeRegularization::paramGrad(const TimeTrajectory& tr,
                                   const HostVector&     prm,
                                   HostVector&           out) const
{
  (void) tr;
  checkParamSize(prm);
  resizeOrZero(out, numParams());

  for (Index i = 0; i < numParams(); ++i)
  {
    out[i] += beta_value_ * (prm[i] - reference_[i]);
  }
  for (Index level = 1; level < num_levels_; ++level)
  {
    for (Index c = 0; c < block_size_; ++c)
    {
      const Real diff           = centered(prm, level, c) - centered(prm, level - 1, c);
      out[index(level, c)]     += beta_difference_ * diff;
      out[index(level - 1, c)] -= beta_difference_ * diff;
    }
  }
}

Index TimeRegularization::index(Index level, Index comp) const
{
  return level * block_size_ + comp;
}

Real TimeRegularization::centered(const HostVector& prm,
                                  Index             level,
                                  Index             comp) const
{
  const Index i = index(level, comp);
  return prm[i] - reference_[i];
}

void TimeRegularization::checkParamSize(const HostVector& prm) const
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("TimeRegularization parameter size mismatch");
  }
}

} // namespace inverse
} // namespace femx
