#include <cmath>
#include <iostream>
#include <stdexcept>

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/state/EnsembleReducedFunctional.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

bool near(Real actual, Real exp, Real tol)
{
  return std::abs(actual - exp) <= tol;
}

Real dot(const Vector<Real>& lhs, const Vector<Real>& rhs)
{
  if (lhs.size() != rhs.size())
  {
    throw std::runtime_error(
        "EnsembleReducedFunctional test dot size mismatch");
  }

  Real value = 0.0;
  for (Index i = 0; i < lhs.size(); ++i)
  {
    value += lhs[i] * rhs[i];
  }
  return value;
}

class QuadraticFunctional final
{
public:
  explicit QuadraticFunctional(Index nprm)
    : target_(nprm),
      weights_(nprm)
  {
    for (Index i = 0; i < nprm; ++i)
    {
      target_[i]  = 0.25 * static_cast<Real>(i + 1);
      weights_[i] = 1.0 + 0.5 * static_cast<Real>(i);
    }
  }

  Index numParams() const
  {
    return target_.size();
  }

  Real value(const Vector<Real>& prm)
  {
    checkPrm(prm);

    Real out = 0.0;
    for (Index i = 0; i < prm.size(); ++i)
    {
      const Real diff  = prm[i] - target_[i];
      out             += 0.5 * weights_[i] * diff * diff;
    }
    return out;
  }

  void grad(const Vector<Real>& prm, Vector<Real>& out)
  {
    checkPrm(prm);

    out.resize(numParams());
    for (Index i = 0; i < prm.size(); ++i)
    {
      out[i] = weights_[i] * (prm[i] - target_[i]);
    }
  }

  Real valueGrad(const Vector<Real>& prm, Vector<Real>& out)
  {
    const Real val = value(prm);
    grad(prm, out);
    return val;
  }

private:
  void checkPrm(const Vector<Real>& prm) const
  {
    if (prm.size() != numParams())
    {
      throw std::runtime_error(
          "QuadraticFunctional parameter size mismatch");
    }
  }

private:
  Vector<Real> target_;
  Vector<Real> weights_;
};

state::EnsembleBasis makeBasis()
{
  Vector<Real> mean{1.0, -2.0, 0.5};
  DenseMatrix  perturbations(3, 2);
  perturbations(0, 0) = 2.0;
  perturbations(1, 0) = 0.0;
  perturbations(2, 0) = -1.0;
  perturbations(0, 1) = -0.5;
  perturbations(1, 1) = 4.0;
  perturbations(2, 1) = 3.0;
  return state::EnsembleBasis(mean, perturbations);
}

} // namespace

class EnsembleReducedFunctionalTests
{
public:
  TestOutcome exposesCoefficientDimension()
  {
    TestStatus status;
    status = true;

    QuadraticFunctional              phys(3);
    state::EnsembleReducedFunctional reduced(phys, makeBasis());

    status *= reduced.numParams() == 2;
    status *= reduced.numPhysicalParams() == 3;
    status *= near(reduced.priorWeight(), 1.0, 0.0);

    return status.report(__func__);
  }

  TestOutcome valueGradMatchesChainRule()
  {
    TestStatus status;
    status = true;

    QuadraticFunctional              phys(3);
    const auto                       basis = makeBasis();
    state::EnsembleReducedFunctional reduced(phys, basis);

    Vector<Real> alpha{0.25, -0.5};
    Vector<Real> prm;
    basis.apply(alpha, prm);

    Vector<Real> phys_grad;
    const Real   phys_value = phys.valueGrad(prm, phys_grad);

    Vector<Real> exp_grad;
    basis.applyT(phys_grad, exp_grad);
    for (Index i = 0; i < alpha.size(); ++i)
    {
      exp_grad[i] += alpha[i];
    }
    const Real exp_value =
        phys_value + 0.5 * dot(alpha, alpha);

    Vector<Real> grad;
    const Real   value = reduced.valueGrad(alpha, grad);

    status *= near(value, exp_value, 1.0e-14);
    status *= grad.size() == exp_grad.size();
    for (Index i = 0; i < grad.size(); ++i)
    {
      status *= near(grad[i], exp_grad[i], 1.0e-14);
    }

    return status.report(__func__);
  }

  TestOutcome gradientMatchesFiniteDifference()
  {
    TestStatus status;
    status = true;

    QuadraticFunctional              phys(3);
    state::EnsembleReducedFunctional reduced(phys, makeBasis());

    Vector<Real> alpha{0.35, -0.2};
    Vector<Real> grad;
    (void) reduced.valueGrad(alpha, grad);

    const Real eps = 1.0e-6;
    for (Index i = 0; i < alpha.size(); ++i)
    {
      Vector<Real> plus   = alpha;
      Vector<Real> minus  = alpha;
      plus[i]            += eps;
      minus[i]           -= eps;

      const Real fd =
          (reduced.value(plus) - reduced.value(minus)) / (2.0 * eps);
      status *= near(grad[i], fd, 1.0e-8);
    }

    return status.report(__func__);
  }

  TestOutcome appliesCustomPriorWeight()
  {
    TestStatus status;
    status = true;

    QuadraticFunctional              phys(3);
    const auto                       basis = makeBasis();
    const Real                       wt    = 0.25;
    state::EnsembleReducedFunctional reduced(phys, basis, wt);

    Vector<Real> alpha{0.25, -0.5};
    Vector<Real> prm;
    basis.apply(alpha, prm);

    Vector<Real> phys_grad;
    const Real   phys_value = phys.valueGrad(prm, phys_grad);

    Vector<Real> exp_grad;
    basis.applyT(phys_grad, exp_grad);
    for (Index i = 0; i < alpha.size(); ++i)
    {
      exp_grad[i] += wt * alpha[i];
    }
    const Real exp_value =
        phys_value + 0.5 * wt * dot(alpha, alpha);

    Vector<Real> grad;
    const Real   value = reduced.valueGrad(alpha, grad);

    status *= near(value, exp_value, 1.0e-14);
    for (Index i = 0; i < grad.size(); ++i)
    {
      status *= near(grad[i], exp_grad[i], 1.0e-14);
    }

    return status.report(__func__);
  }

  TestOutcome rejectsInvalidInputs()
  {
    TestStatus status;
    status = true;

    bool threw = false;
    try
    {
      QuadraticFunctional              phys(4);
      state::EnsembleReducedFunctional reduced(phys, makeBasis());
      (void) reduced;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    threw = false;
    try
    {
      QuadraticFunctional              phys(3);
      state::EnsembleReducedFunctional reduced(phys, makeBasis());
      Vector<Real>                     alpha{1.0};
      Vector<Real>                     grad;
      (void) reduced.valueGrad(alpha, grad);
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    threw = false;
    try
    {
      QuadraticFunctional              phys(3);
      state::EnsembleReducedFunctional reduced(phys, makeBasis(), -1.0);
      (void) reduced;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults                 results;
  femx::tests::EnsembleReducedFunctionalTests test;

  results += test.exposesCoefficientDimension();
  results += test.valueGradMatchesChainRule();
  results += test.gradientMatchesFiniteDifference();
  results += test.appliesCustomPriorWeight();
  results += test.rejectsInvalidInputs();

  return results.summary();
}
