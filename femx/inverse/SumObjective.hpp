#pragma once

#include <femx/common/Types.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Sum of stationary objective terms with the same dimensions.
 *
 * SumObjective stores references to compatible terms and accumulates their
 * values, state gradients, and parameter gradients.
 */
class SumObjective final : public Objective
{
public:
  SumObjective(Index num_states, Index num_params);

  SumObjective& add(const Objective& term);

  Index numStates() const override;
  Index numParams() const override;

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override;

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override;

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override;

private:
  static void addInto(const Vector<Real>& input,
                      Vector<Real>&       out,
                      Index               size);

private:
  Index                    num_states_{0};
  Index                    num_params_{0};
  Vector<const Objective*> terms_;
};

} // namespace inverse
} // namespace femx
