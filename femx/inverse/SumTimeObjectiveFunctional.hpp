#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Non-owning sum of time-dependent objective terms. */
class SumTimeObjectiveFunctional final : public TimeObjectiveFunctional
{
public:
  SumTimeObjectiveFunctional(Index num_steps,
                             Index num_states,
                             Index num_prm);

  SumTimeObjectiveFunctional& add(const TimeObjectiveFunctional& term);

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
  static void resize(Vector<Real>& out,
                     Index         size);

  static void addInto(const Vector<Real>& input,
                      Vector<Real>&       out,
                      Index               size);

private:
  Index num_steps_{0};
  Index num_states_{0};
  Index num_prm_{0};

  std::vector<const TimeObjectiveFunctional*> terms_;
};

} // namespace inverse
} // namespace femx
