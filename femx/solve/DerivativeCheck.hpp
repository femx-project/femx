#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/ResidualEquation.hpp>
#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/solve/ReducedObjective.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace solve
{

using problem::ObjectiveFunctional;

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

  DerivativeCheckResult objStateGrad(const ObjectiveFunctional& obj,
                                     const Vector<Real>&        state,
                                     const Vector<Real>&        prm,
                                     const Vector<Real>&        dir) const;

  DerivativeCheckResult objParamGrad(const ObjectiveFunctional& obj,
                                     const Vector<Real>&        state,
                                     const Vector<Real>&        prm,
                                     const Vector<Real>&        dir) const;

  DerivativeCheckResult reducedGrad(ReducedObjective&   functional,
                                    const Vector<Real>& prm,
                                    const Vector<Real>& dir) const;

  DerivativeCheckResult resStateJac(const problem::ResidualEquation& eq,
                                    const Vector<Real>&         state,
                                    const Vector<Real>&         prm,
                                    const Vector<Real>&         dir) const;

  DerivativeCheckResult resParamJac(const problem::ResidualEquation& eq,
                                    const Vector<Real>&         state,
                                    const Vector<Real>&         prm,
                                    const Vector<Real>&         dir) const;

  DerivativeCheckResult stateJacT(const problem::ResidualEquation& eq,
                                  const Vector<Real>&         state,
                                  const Vector<Real>&         prm,
                                  const Vector<Real>&         dir,
                                  const Vector<Real>&         lambda) const;

  DerivativeCheckResult paramJacT(const problem::ResidualEquation& eq,
                                  const Vector<Real>&         state,
                                  const Vector<Real>&         prm,
                                  const Vector<Real>&         dir,
                                  const Vector<Real>&         lambda) const;

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

} // namespace solve
} // namespace femx
