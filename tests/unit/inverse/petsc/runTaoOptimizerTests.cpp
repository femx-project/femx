#include <petsctao.h>

#include <cmath>
#include <iostream>

#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/petsc/TaoOptimizer.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class QuadraticReducedFunctional final : public inverse::ReducedFunctional
{
public:
  Index numParams() const override
  {
    return 2;
  }

  Real value(const Vector<Real>& params) override
  {
    return 0.5 * (params[0] - 1.0) * (params[0] - 1.0)
           + (params[1] + 2.0) * (params[1] + 2.0);
  }

  void grad(const Vector<Real>& params, Vector<Real>& out) override
  {
    if (out.size() != numParams())
    {
      out.resize(numParams());
    }
    else
    {
      out.setZero();
    }

    out[0] = params[0] - 1.0;
    out[1] = 2.0 * (params[1] + 2.0);
  }
};

class TaoOptimizerTests : public TestBase
{
public:
  TestOutcome solveMinimizesQuadratic()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional functional;
    inverse::TaoOptimizer      optimizer(functional);
    optimizer.options().grad_abs_tolerance = 1.0e-12;
    optimizer.options().grad_rel_tolerance = 1.0e-12;
    optimizer.options().max_its            = 50;
    optimizer.options().use_opts_db        = false;

    Vector<Real> initial(2);
    initial[0] = 0.25;
    initial[1] = -0.5;

    inverse::TaoResult   result;
    const PetscErrorCode ierr = optimizer.solve(initial, result);

    status *= (ierr == PETSC_SUCCESS);
    status *= result.converged();
    status *= (result.value < 1.0e-20);
    status *= (result.grad_norm_squared < 1.0e-20);
    status *= (std::abs(result.params[0] - 1.0) < 1.0e-10);
    status *= (std::abs(result.params[1] + 2.0) < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome solveRejectsWrongInitialSize()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional functional;
    inverse::TaoOptimizer      optimizer(functional);

    Vector<Real> initial(1);
    initial[0] = 0.25;

    inverse::TaoResult   result;
    const PetscErrorCode ierr = optimizer.solve(initial, result);

    status *= (ierr == PETSC_ERR_ARG_SIZ);

    return status.report(__func__);
  }

  TestOutcome solveRespectsBoxBounds()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional functional;
    inverse::TaoOptimizer      optimizer(functional);
    optimizer.options().grad_abs_tolerance = 1.0e-12;
    optimizer.options().grad_rel_tolerance = 1.0e-12;
    optimizer.options().max_its            = 50;
    optimizer.options().use_opts_db        = false;

    Vector<Real> lower(2);
    lower[0] = 0.0;
    lower[1] = -1.0;

    Vector<Real> upper(2);
    upper[0] = 0.5;
    upper[1] = 0.0;

    optimizer.setBounds(lower, upper);

    Vector<Real> initial(2);
    initial[0] = 0.25;
    initial[1] = -0.5;

    inverse::TaoResult   result;
    const PetscErrorCode ierr = optimizer.solve(initial, result);

    status *= (ierr == PETSC_SUCCESS);
    status *= result.converged();
    status *= (std::abs(result.params[0] - 0.5) < 1.0e-10);
    status *= (std::abs(result.params[1] + 1.0) < 1.0e-10);
    status *= (std::abs(result.value - 1.125) < 1.0e-10);

    return status.report(__func__);
  }

  TestOutcome solveRejectsWrongBoundsSize()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional functional;
    inverse::TaoOptimizer      optimizer(functional);

    Vector<Real> lower(1);
    lower[0] = 0.0;

    Vector<Real> upper(2);
    upper[0] = 1.0;
    upper[1] = 1.0;

    optimizer.setBounds(lower, upper);

    Vector<Real> initial(2);
    initial[0] = 0.25;
    initial[1] = -0.5;

    inverse::TaoResult   result;
    const PetscErrorCode ierr = optimizer.solve(initial, result);

    status *= (ierr == PETSC_ERR_ARG_SIZ);

    return status.report(__func__);
  }

  TestOutcome solveRejectsInvertedBounds()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional functional;
    inverse::TaoOptimizer      optimizer(functional);

    Vector<Real> lower(2);
    lower[0] = 2.0;
    lower[1] = -1.0;

    Vector<Real> upper(2);
    upper[0] = 1.0;
    upper[1] = 1.0;

    optimizer.setBounds(lower, upper);

    Vector<Real> initial(2);
    initial[0] = 0.25;
    initial[1] = -0.5;

    inverse::TaoResult   result;
    const PetscErrorCode ierr = optimizer.solve(initial, result);

    status *= (ierr == PETSC_ERR_ARG_OUTOFRANGE);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  std::cout << "Running TAO optimizer tests:\n";

  femx::tests::TaoOptimizerTests test;

  femx::tests::TestingResults result;
  result += test.solveMinimizesQuadratic();
  result += test.solveRejectsWrongInitialSize();
  result += test.solveRespectsBoxBounds();
  result += test.solveRejectsWrongBoundsSize();
  result += test.solveRejectsInvertedBounds();

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  return result.summary();
}
