#pragma once

#include <femx/common/Types.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

/** Sparse block quadratic regularization repeated over control levels. */
class TimeBlockRegularization final : public TimeObjective
{
public:
  TimeBlockRegularization(Index               num_steps,
                          Index               num_states,
                          Index               num_levels,
                          Index               block_size,
                          Vector<Index>       rows,
                          Vector<Index>       cols,
                          Vector<Real>        vals,
                          Real                weight,
                          const Vector<Real>& reference = {});

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
  Index index(Index level, Index comp) const;
  Real  centered(const Vector<Real>& prm, Index level, Index comp) const;
  void  checkParamSize(const Vector<Real>& prm) const;

private:
  Index         num_steps_{0};
  Index         num_states_{0};
  Index         num_levels_{0};
  Index         block_size_{0};
  Vector<Index> rows_;
  Vector<Index> cols_;
  Vector<Real>  vals_;
  Real          weight_{0.0};
  Vector<Real>  reference_;
};

} // namespace inverse
} // namespace femx
