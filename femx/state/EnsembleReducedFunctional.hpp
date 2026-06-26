#pragma once

#include <functional>
#include <stdexcept>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/EnsembleBasis.hpp>

namespace femx
{
namespace state
{

/** @brief Reduced functional over ensemble coefficients alpha.
 *
 * The wrapped physical functional is evaluated at
 * prm = mean + perturbations * alpha. The ensemble-space objective adds the
 * Gaussian prior 0.5 * prior_weight * ||alpha||^2.
 */
class EnsembleReducedFunctional final
{
public:
  using ValueCallback =
      std::function<Real(const Vector<Real>& prm)>;
  using ValueGradCallback =
      std::function<Real(const Vector<Real>& prm,
                         Vector<Real>&       grad_out)>;

  EnsembleReducedFunctional(EnsembleBasis     basis,
                            ValueCallback     value,
                            ValueGradCallback value_grad,
                            Real              prior_weight = 1.0);

  template <typename Functional,
            typename = decltype(std::declval<Functional&>().numParams()),
            typename = decltype(std::declval<Functional&>().value(
                std::declval<const Vector<Real>&>())),
            typename = decltype(std::declval<Functional&>().valueGrad(
                std::declval<const Vector<Real>&>(),
                std::declval<Vector<Real>&>()))>
  EnsembleReducedFunctional(Functional&   fn,
                            EnsembleBasis basis,
                            Real          prior_weight = 1.0)
    : EnsembleReducedFunctional(
          std::move(basis),
          [&fn](const Vector<Real>& prm)
          { return fn.value(prm); },
          [&fn](const Vector<Real>& prm, Vector<Real>& grad)
          { return fn.valueGrad(prm, grad); },
          prior_weight)
  {
    if (fn.numParams() != basis_.numPhysicalParams())
    {
      throw std::runtime_error(
          "EnsembleReducedFunctional physical parameter size mismatch");
    }
  }

  Index numParams() const;
  Index numPhysicalParams() const;

  Real                 priorWeight() const;
  const EnsembleBasis& basis() const;

  Real value(const Vector<Real>& alpha);
  void grad(const Vector<Real>& alpha, Vector<Real>& out);
  Real valueGrad(const Vector<Real>& alpha, Vector<Real>& grad_out);

private:
  void checkReady() const;
  void checkAlpha(const Vector<Real>& alpha) const;
  void checkPhysicalGrad(const Vector<Real>& grad) const;

private:
  EnsembleBasis     basis_;
  ValueCallback     value_;
  ValueGradCallback value_grad_;
  Real              prior_weight_{1.0};
};

} // namespace state
} // namespace femx
