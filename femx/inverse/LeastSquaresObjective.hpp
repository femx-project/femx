#pragma once

#include <femx/common/Types.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Objective 0.5 * weight * ||H(u,m) - data||^2. */
class LeastSquaresObjective final : public ObjectiveFunctional
{
public:
  LeastSquaresObjective(const ObservationOperator& obs,
                        const Vector<Real>&        data,
                        Real                       weight = 1.0);

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
  const ObservationOperator& obs_;
  Vector<Real>               data_;
  Real                       weight_{1.0};
};

} // namespace inverse
} // namespace femx
