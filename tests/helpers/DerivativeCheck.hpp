#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Linearization.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/state/Residual.hpp>
#include <femx/inverse/ReducedFunctional.hpp>

namespace femx
{
namespace tests
{

struct DerivativeCheckResult
{
  Real computed  = 0.0;
  Real reference = 0.0;
  Real abs_err   = 0.0;
  Real rel_err   = 0.0;

  bool passed(Real atol,
              Real rtol) const;
};

class DerivativeCheck
{
public:
  explicit DerivativeCheck(Real step = 1.0e-6);

  DerivativeCheckResult objectiveStateGrad(const inverse::Objective& obj,
                                           const Vector<Real>&       state,
                                           const Vector<Real>&       prm,
                                           const Vector<Real>&       dir) const;

  DerivativeCheckResult objectiveParamGrad(const inverse::Objective& obj,
                                           const Vector<Real>&       state,
                                           const Vector<Real>&       prm,
                                           const Vector<Real>&       dir) const;

  DerivativeCheckResult reducedGrad(inverse::ReducedFunctional& fn,
                                    const Vector<Real>&       prm,
                                    const Vector<Real>&       dir) const;

  DerivativeCheckResult residualStateJacobian(
      const state::Residual& problem,
      state::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir) const;

  DerivativeCheckResult residualParamJacobian(
      const state::Residual& problem,
      state::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir) const;

  DerivativeCheckResult stateJacTranspose(
      const state::Residual& problem,
      state::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir,
      const Vector<Real>&      lambda) const;

  DerivativeCheckResult paramJacTranspose(
      const state::Residual& problem,
      state::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir,
      const Vector<Real>&      lambda) const;

private:
  Vector<Real> shifted(const Vector<Real>& base,
                       const Vector<Real>& dir,
                       Real                scale) const;

  Vector<Real> centralDifference(const Vector<Real>& plus,
                                 const Vector<Real>& minus) const;

  static DerivativeCheckResult compareScalars(Real computed,
                                              Real reference);

  static DerivativeCheckResult compareVectors(const Vector<Real>& computed,
                                              const Vector<Real>& reference);

private:
  Real step_{1.0e-6};
};

} // namespace tests
} // namespace femx
