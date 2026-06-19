#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Non-owning sum of objective terms with matching dimensions. */
class SumObjectiveFunctional final : public ObjectiveFunctional
{
public:
  SumObjectiveFunctional(Index num_states,
                         Index num_prm);

  SumObjectiveFunctional& add(const ObjectiveFunctional& term);

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
  static void resize(Vector<Real>& out, Index size);

  static void addInto(const Vector<Real>& input, Vector<Real>& out, Index size);

private:
  Index num_states_{0};
  Index num_prm_{0};

  std::vector<const ObjectiveFunctional*> terms_;
};

} // namespace inverse
} // namespace femx
