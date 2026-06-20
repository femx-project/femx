#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace problem
{

/** @brief Objective 0.5 * weight * ||m - reference||^2. */
class QuadraticParameterRegularization final : public ObjectiveFunctional
{
public:
  QuadraticParameterRegularization(Index               num_states,
                                   const Vector<Real>& reference,
                                   Real                weight);

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
  void checkParams(const Vector<Real>& prm) const;

  static void resize(Vector<Real>& out, Index size);

private:
  Index        num_states_{0};
  Vector<Real> reference_;
  Real         weight_{0.0};
};

} // namespace problem
} // namespace femx
