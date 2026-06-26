#include <cmath>
#include <iostream>
#include <stdexcept>

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/state/EnsembleBasis.hpp>
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
    throw std::runtime_error("EnsembleBasis test dot size mismatch");
  }

  Real value = 0.0;
  for (Index i = 0; i < lhs.size(); ++i)
  {
    value += lhs[i] * rhs[i];
  }
  return value;
}

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

class EnsembleBasisTests
{
public:
  TestOutcome applyMapsCoefficientsToPhysicalParams()
  {
    TestStatus status;
    status = true;

    const state::EnsembleBasis basis = makeBasis();

    Vector<Real> alpha{0.25, -0.5};
    Vector<Real> prm;
    basis.apply(alpha, prm);

    status *= prm.size() == 3;
    status *= near(prm[0], 1.75, 1.0e-14);
    status *= near(prm[1], -4.0, 1.0e-14);
    status *= near(prm[2], -1.25, 1.0e-14);

    return status.report(__func__);
  }

  TestOutcome applyTMapsPhysicalGradientToCoefficients()
  {
    TestStatus status;
    status = true;

    const state::EnsembleBasis basis = makeBasis();

    Vector<Real> grad{1.5, -2.0, 0.25};
    Vector<Real> coeff_grad;
    basis.applyT(grad, coeff_grad);

    status *= coeff_grad.size() == 2;
    status *= near(coeff_grad[0], 2.75, 1.0e-14);
    status *= near(coeff_grad[1], -8.0, 1.0e-14);

    return status.report(__func__);
  }

  TestOutcome applyTIsAdjointOfPerturbationMap()
  {
    TestStatus status;
    status = true;

    const state::EnsembleBasis basis = makeBasis();

    Vector<Real> alpha{0.35, -0.2};
    Vector<Real> prm;
    basis.apply(alpha, prm);

    Vector<Real> inc(prm.size());
    for (Index i = 0; i < inc.size(); ++i)
    {
      inc[i] = prm[i] - basis.mean()[i];
    }

    Vector<Real> grad{0.8, -1.4, 2.1};
    Vector<Real> coeff_grad;
    basis.applyT(grad, coeff_grad);

    const Real lhs  = dot(inc, grad);
    const Real rhs  = dot(alpha, coeff_grad);
    status         *= near(lhs, rhs, 1.0e-14);

    return status.report(__func__);
  }

  TestOutcome allowsAliasedOutput()
  {
    TestStatus status;
    status = true;

    const state::EnsembleBasis basis = makeBasis();

    Vector<Real> alpha{0.25, -0.5};
    basis.apply(alpha, alpha);
    status *= alpha.size() == 3;
    status *= near(alpha[0], 1.75, 1.0e-14);
    status *= near(alpha[1], -4.0, 1.0e-14);
    status *= near(alpha[2], -1.25, 1.0e-14);

    Vector<Real> grad{1.5, -2.0, 0.25};
    basis.applyT(grad, grad);
    status *= grad.size() == 2;
    status *= near(grad[0], 2.75, 1.0e-14);
    status *= near(grad[1], -8.0, 1.0e-14);

    return status.report(__func__);
  }

  TestOutcome rejectsInconsistentDimensions()
  {
    TestStatus status;
    status = true;

    bool threw = false;
    try
    {
      Vector<Real>         mean{1.0, 2.0};
      DenseMatrix          perturbations(3, 1);
      state::EnsembleBasis basis(mean, perturbations);
      (void) basis;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    const state::EnsembleBasis basis = makeBasis();
    threw                            = false;
    try
    {
      Vector<Real> alpha{1.0};
      Vector<Real> out;
      basis.apply(alpha, out);
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }
    status *= threw;

    threw = false;
    try
    {
      Vector<Real> grad{1.0, 2.0};
      Vector<Real> out;
      basis.applyT(grad, out);
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
  femx::tests::TestingResults     results;
  femx::tests::EnsembleBasisTests test;

  results += test.applyMapsCoefficientsToPhysicalParams();
  results += test.applyTMapsPhysicalGradientToCoefficients();
  results += test.applyTIsAdjointOfPerturbationMap();
  results += test.allowsAliasedOutput();
  results += test.rejectsInconsistentDimensions();

  return results.summary();
}
