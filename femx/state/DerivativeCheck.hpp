#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/Linearization.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/state/ReducedFunctional.hpp>

namespace femx
{
namespace state
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

  DerivativeCheckResult objectiveStateGrad(const problem::Objective& obj,
                                           const Vector<Real>&       state,
                                           const Vector<Real>&       prm,
                                           const Vector<Real>&       dir) const;

  DerivativeCheckResult objectiveParamGrad(const problem::Objective& obj,
                                           const Vector<Real>&       state,
                                           const Vector<Real>&       prm,
                                           const Vector<Real>&       dir) const;

  DerivativeCheckResult reducedGrad(ReducedFunctional&  fn,
                                    const Vector<Real>& prm,
                                    const Vector<Real>& dir) const;

  DerivativeCheckResult residualStateJacobian(
      const problem::Residual& problem,
      problem::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir) const;

  DerivativeCheckResult residualParamJacobian(
      const problem::Residual& problem,
      problem::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir) const;

  DerivativeCheckResult stateJacTranspose(
      const problem::Residual& problem,
      problem::Linearization&  lin,
      const Vector<Real>&      state,
      const Vector<Real>&      prm,
      const Vector<Real>&      dir,
      const Vector<Real>&      lambda) const;

  DerivativeCheckResult paramJacTranspose(
      const problem::Residual& problem,
      problem::Linearization&  lin,
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

} // namespace state
} // namespace femx
