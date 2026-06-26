#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/state/EnsembleReducedFunctional.hpp>

using namespace std;

namespace femx
{
namespace state
{

namespace
{

Real squaredNorm(const Vector<Real>& x)
{
  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    value += x[i] * x[i];
  }
  return value;
}

} // namespace

EnsembleReducedFunctional::EnsembleReducedFunctional(
    EnsembleBasis     basis,
    ValueCallback     value,
    ValueGradCallback value_grad,
    Real              prior_weight)
  : basis_(std::move(basis)),
    value_(std::move(value)),
    value_grad_(std::move(value_grad)),
    prior_weight_(prior_weight)
{
  checkReady();
}

Index EnsembleReducedFunctional::numParams() const
{
  return basis_.numCoefficients();
}

Index EnsembleReducedFunctional::numPhysicalParams() const
{
  return basis_.numPhysicalParams();
}

Real EnsembleReducedFunctional::priorWeight() const
{
  return prior_weight_;
}

const EnsembleBasis& EnsembleReducedFunctional::basis() const
{
  return basis_;
}

Real EnsembleReducedFunctional::value(const Vector<Real>& alpha)
{
  checkAlpha(alpha);

  Vector<Real> prm;
  basis_.apply(alpha, prm);
  return value_(prm) + 0.5 * prior_weight_ * squaredNorm(alpha);
}

void EnsembleReducedFunctional::grad(const Vector<Real>& alpha,
                                     Vector<Real>&       out)
{
  (void) valueGrad(alpha, out);
}

Real EnsembleReducedFunctional::valueGrad(const Vector<Real>& alpha,
                                          Vector<Real>&       grad_out)
{
  checkAlpha(alpha);

  Vector<Real> prm;
  basis_.apply(alpha, prm);

  Vector<Real> phys_grad;
  const Real   phys_value = value_grad_(prm, phys_grad);
  checkPhysicalGrad(phys_grad);

  Vector<Real> coeff_grad;
  basis_.applyT(phys_grad, coeff_grad);
  for (Index i = 0; i < coeff_grad.size(); ++i)
  {
    coeff_grad[i] += prior_weight_ * alpha[i];
  }

  grad_out = std::move(coeff_grad);
  return phys_value + 0.5 * prior_weight_ * squaredNorm(alpha);
}

void EnsembleReducedFunctional::checkReady() const
{
  if (!value_ || !value_grad_)
  {
    throw runtime_error(
        "EnsembleReducedFunctional requires value callbacks");
  }
  if (!std::isfinite(prior_weight_) || prior_weight_ < 0.0)
  {
    throw runtime_error(
        "EnsembleReducedFunctional received invalid prior weight");
  }
  (void) basis_.numPhysicalParams();
  (void) basis_.numCoefficients();
}

void EnsembleReducedFunctional::checkAlpha(
    const Vector<Real>& alpha) const
{
  checkReady();
  if (alpha.size() != numParams())
  {
    throw runtime_error(
        "EnsembleReducedFunctional coefficient size mismatch");
  }
}

void EnsembleReducedFunctional::checkPhysicalGrad(
    const Vector<Real>& grad) const
{
  if (grad.size() != numPhysicalParams())
  {
    throw runtime_error(
        "EnsembleReducedFunctional physical gradient size mismatch");
  }
}

} // namespace state
} // namespace femx
