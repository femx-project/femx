#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
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

  DerivativeCheckResult objStateGrad(const ObjectiveFunctional& obj,
                                     const Vector<Real>&        state,
                                     const Vector<Real>&        prm,
                                     const Vector<Real>&        dir) const;

  DerivativeCheckResult objParamGrad(const ObjectiveFunctional& obj,
                                     const Vector<Real>&        state,
                                     const Vector<Real>&        prm,
                                     const Vector<Real>&        dir) const;

  DerivativeCheckResult reducedGrad(ReducedFunctional&  functional,
                                    const Vector<Real>& prm,
                                    const Vector<Real>& dir) const;

  DerivativeCheckResult resStateJac(const eq::ResidualEquation& eq,
                                    const Vector<Real>&         state,
                                    const Vector<Real>&         prm,
                                    const Vector<Real>&         dir) const;

  DerivativeCheckResult resParamJac(const eq::ResidualEquation& eq,
                                    const Vector<Real>&         state,
                                    const Vector<Real>&         prm,
                                    const Vector<Real>&         dir) const;

  DerivativeCheckResult stateJacT(const eq::ResidualEquation& eq,
                                  const Vector<Real>&         state,
                                  const Vector<Real>&         prm,
                                  const Vector<Real>&         dir,
                                  const Vector<Real>&         lambda) const;

  DerivativeCheckResult paramJacT(const eq::ResidualEquation& eq,
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

} // namespace inverse
} // namespace femx
