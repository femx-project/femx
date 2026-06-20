#include <petsctao.h>

#include <iostream>

#include <femx/solve/ReducedObjective.hpp>
#include <femx/optimize/TaoReducedFunctionalAdapter.hpp>
#include <femx/algebra/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class QuadraticReducedFunctional final : public solve::ReducedObjective
{
public:
  Index numParams() const override
  {
    return 2;
  }

  Real value(const Vector<Real>& prm) override
  {
    return 0.5 * (prm[0] - 1.0) * (prm[0] - 1.0)
           + (prm[1] + 2.0) * (prm[1] + 2.0);
  }

  void grad(const Vector<Real>& prm, Vector<Real>& out) override
  {
    if (out.size() != numParams())
    {
      out.resize(numParams());
    }
    else
    {
      out.setZero();
    }

    out[0] = prm[0] - 1.0;
    out[1] = 2.0 * (prm[1] + 2.0);
  }
};

class TaoReducedFunctionalAdapterTests : public TestBase
{
public:
  TestOutcome callbackEvaluatesValueAndGradient()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional           functional;
    optimize::TaoReducedFunctionalAdapter adapter(functional);

    Vec prm  = nullptr;
    Vec grad = nullptr;
    Tao tao  = nullptr;

    PetscErrorCode ierr  = VecCreateSeq(PETSC_COMM_SELF, 2, &prm);
    status              *= (ierr == 0);
    ierr                 = VecDuplicate(prm, &grad);
    status              *= (ierr == 0);
    ierr                 = TaoCreate(PETSC_COMM_SELF, &tao);
    status              *= (ierr == 0);

    const PetscInt    indices[2]  = {0, 1};
    const PetscScalar values[2]   = {0.25, -0.5};
    ierr                          = VecSetValues(prm, 2, indices, values, INSERT_VALUES);
    status                       *= (ierr == 0);
    ierr                          = VecAssemblyBegin(prm);
    status                       *= (ierr == 0);
    ierr                          = VecAssemblyEnd(prm);
    status                       *= (ierr == 0);

    PetscReal value = 0.0;
    ierr            = optimize::TaoReducedFunctionalAdapter::
        formValueGrad(tao, prm, &value, grad, &adapter);
    status *= (ierr == 0);

    status *= isEqual(value,
                      0.5 * (values[0] - 1.0) * (values[0] - 1.0)
                          + (values[1] + 2.0) * (values[1] + 2.0));

    const PetscScalar* grad_values  = nullptr;
    ierr                            = VecGetArrayRead(grad, &grad_values);
    status                         *= (ierr == 0);
    status                         *= isEqual(PetscRealPart(grad_values[0]),
                      values[0] - 1.0);
    status                         *= isEqual(PetscRealPart(grad_values[1]),
                      2.0 * (values[1] + 2.0));
    ierr                            = VecRestoreArrayRead(grad, &grad_values);
    status                         *= (ierr == 0);

    if (tao != nullptr)
    {
      TaoDestroy(&tao);
    }
    if (grad != nullptr)
    {
      VecDestroy(&grad);
    }
    if (prm != nullptr)
    {
      VecDestroy(&prm);
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != 0)
  {
    return 1;
  }

  std::cout << "Running TAO reduced functional adapter tests:\n";

  femx::tests::TaoReducedFunctionalAdapterTests test;

  femx::tests::TestingResults result;
  result += test.callbackEvaluatesValueAndGradient();

  ierr = PetscFinalize();
  if (ierr != 0)
  {
    return 1;
  }

  return result.summary();
}
