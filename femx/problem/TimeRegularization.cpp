#include <stdexcept>

#include <femx/problem/TimeRegularization.hpp>

using namespace std;
using namespace femx::state;

namespace femx
{
namespace problem
{

TimeRegularization::TimeRegularization(Index               nt,
                                       Index               nst,
                                       Index               nl,
                                       Index               block_size,
                                       Real                beta_difference,
                                       Real                beta_value,
                                       const Vector<Real>& reference)
  : nt_(nt),
    nst_(nst),
    nl_(nl),
    block_size_(block_size),
    beta_difference_(beta_difference),
    beta_value_(beta_value)
{
  if (nt_ < 0 || nst_ < 0 || nl_ < 0
      || block_size_ < 0 || beta_difference_ < 0.0 || beta_value_ < 0.0)
  {
    throw runtime_error("TimeRegularization received invalid inputs");
  }
  if (reference.empty())
  {
    reference_.resize(numParams());
  }
  else if (reference.size() == numParams())
  {
    reference_ = reference;
  }
  else
  {
    throw runtime_error("TimeRegularization reference size mismatch");
  }
}

Index TimeRegularization::numSteps() const
{
  return nt_;
}

Index TimeRegularization::numStates() const
{
  return nst_;
}

Index TimeRegularization::numParams() const
{
  return nl_ * block_size_;
}

Real TimeRegularization::value(const TimeTrajectory& tr,
                               const Vector<Real>&   prm) const
{
  (void) tr;
  checkParamSize(prm);

  Real value_out = 0.0;
  for (Index i = 0; i < numParams(); ++i)
  {
    const Real diff  = prm[i] - reference_[i];
    value_out       += 0.5 * beta_value_ * diff * diff;
  }
  for (Index level = 1; level < nl_; ++level)
  {
    for (Index c = 0; c < block_size_; ++c)
    {
      const Real diff =
          centered(prm, level, c) - centered(prm, level - 1, c);
      value_out += 0.5 * beta_difference_ * diff * diff;
    }
  }
  return value_out;
}

void TimeRegularization::stateGrad(Index                 level,
                                   const TimeTrajectory& tr,
                                   const Vector<Real>&   prm,
                                   Vector<Real>&         out) const
{
  (void) level;
  (void) tr;
  (void) prm;
  resizeOrZero(out, numStates());
}

void TimeRegularization::paramGrad(const TimeTrajectory& tr,
                                   const Vector<Real>&   prm,
                                   Vector<Real>&         out) const
{
  (void) tr;
  checkParamSize(prm);
  resizeOrZero(out, numParams());

  for (Index i = 0; i < numParams(); ++i)
  {
    out[i] += beta_value_ * (prm[i] - reference_[i]);
  }
  for (Index level = 1; level < nl_; ++level)
  {
    for (Index c = 0; c < block_size_; ++c)
    {
      const Real diff =
          centered(prm, level, c) - centered(prm, level - 1, c);
      out[index(level, c)]     += beta_difference_ * diff;
      out[index(level - 1, c)] -= beta_difference_ * diff;
    }
  }
}

Index TimeRegularization::index(Index level, Index comp) const
{
  return level * block_size_ + comp;
}

Real TimeRegularization::centered(const Vector<Real>& prm,
                                  Index               level,
                                  Index               comp) const
{
  const Index i = index(level, comp);
  return prm[i] - reference_[i];
}

void TimeRegularization::checkParamSize(const Vector<Real>& prm) const
{
  if (prm.size() != numParams())
  {
    throw runtime_error("TimeRegularization parameter size mismatch");
  }
}

} // namespace problem
} // namespace femx
