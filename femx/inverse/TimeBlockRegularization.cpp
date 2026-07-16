#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/inverse/TimeBlockRegularization.hpp>
using namespace femx::state;

namespace femx
{
namespace inverse
{

TimeBlockRegularization::TimeBlockRegularization(
    Index               num_steps,
    Index               num_states,
    Index               num_levels,
    Index               block_size,
    Vector<Index>       rows,
    Vector<Index>       cols,
    Vector<Real>        vals,
    Real                weight,
    const Vector<Real>& reference)
  : num_steps_(num_steps),
    num_states_(num_states),
    num_levels_(num_levels),
    block_size_(block_size),
    rows_(std::move(rows)),
    cols_(std::move(cols)),
    vals_(std::move(vals)),
    weight_(weight)
{
  if (num_steps_ < 0 || num_states_ < 0 || num_levels_ < 0
      || block_size_ < 0 || !std::isfinite(weight_) || weight_ < 0.0
      || rows_.size() != cols_.size() || rows_.size() != vals_.size())
  {
    throw std::runtime_error(
        "TimeBlockRegularization received invalid inputs");
  }
  if (weight_ > 0.0 && vals_.empty())
  {
    throw std::runtime_error(
        "TimeBlockRegularization received an empty matrix");
  }
  for (Index i = 0; i < vals_.size(); ++i)
  {
    if (rows_[i] < 0 || rows_[i] >= block_size_
        || cols_[i] < 0 || cols_[i] >= block_size_
        || !std::isfinite(vals_[i]))
    {
      throw std::runtime_error(
          "TimeBlockRegularization received an invalid matrix");
    }
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
            "TimeBlockRegularization reference must be finite");
      }
    }
    reference_ = reference;
  }
  else
  {
    throw std::runtime_error(
        "TimeBlockRegularization reference size mismatch");
  }
}

Index TimeBlockRegularization::numSteps() const
{
  return num_steps_;
}

Index TimeBlockRegularization::numStates() const
{
  return num_states_;
}

Index TimeBlockRegularization::numParams() const
{
  return num_levels_ * block_size_;
}

Real TimeBlockRegularization::value(const TimeTrajectory& tr,
                                    const Vector<Real>&   prm) const
{
  (void) tr;
  checkParamSize(prm);

  Real out = 0.0;
  for (Index level = 0; level < num_levels_; ++level)
  {
    for (Index i = 0; i < vals_.size(); ++i)
    {
      out += 0.5 * weight_ * vals_[i]
             * centered(prm, level, rows_[i])
             * centered(prm, level, cols_[i]);
    }
  }
  return out;
}

void TimeBlockRegularization::stateGrad(Index                 level,
                                        const TimeTrajectory& tr,
                                        const Vector<Real>&   prm,
                                        Vector<Real>&         out) const
{
  (void) level;
  (void) tr;
  (void) prm;
  resizeOrZero(out, numStates());
}

void TimeBlockRegularization::paramGrad(const TimeTrajectory& tr,
                                        const Vector<Real>&   prm,
                                        Vector<Real>&         out) const
{
  (void) tr;
  checkParamSize(prm);
  resizeOrZero(out, numParams());

  for (Index level = 0; level < num_levels_; ++level)
  {
    for (Index i = 0; i < vals_.size(); ++i)
    {
      const Index row = index(level, rows_[i]);
      const Index col = index(level, cols_[i]);
      const Real  val = 0.5 * weight_ * vals_[i];
      out[row] += val * centered(prm, level, cols_[i]);
      out[col] += val * centered(prm, level, rows_[i]);
    }
  }
}

Index TimeBlockRegularization::index(Index level, Index comp) const
{
  return level * block_size_ + comp;
}

Real TimeBlockRegularization::centered(const Vector<Real>& prm,
                                       Index               level,
                                       Index               comp) const
{
  const Index i = index(level, comp);
  return prm[i] - reference_[i];
}

void TimeBlockRegularization::checkParamSize(
    const Vector<Real>& prm) const
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeBlockRegularization parameter size mismatch");
  }
}

} // namespace inverse
} // namespace femx
