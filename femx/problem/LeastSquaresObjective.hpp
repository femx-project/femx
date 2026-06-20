#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/problem/Observation.hpp>

namespace femx
{
namespace problem
{

/** @brief Objective 0.5 * weight * ||H(u,m) - data||^2. */
class LeastSquaresObjective final : public ObjectiveFunctional
{
public:
  LeastSquaresObjective(const problem::Observation& obs,
                        const Vector<Real>&         data,
                        Real                        weight = 1.0);

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
  void obsResidual(const Vector<Real>& state,
                   const Vector<Real>& prm,
                   Vector<Real>&       out) const;

  static void scale(Vector<Real>& out, Real factor);

private:
  const problem::Observation& obs_;
  Vector<Real>                data_;
  Real                        weight_{1.0};
};

} // namespace problem
} // namespace femx
