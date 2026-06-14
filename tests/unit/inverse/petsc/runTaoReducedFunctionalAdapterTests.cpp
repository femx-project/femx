#include <petsctao.h>

#include <iostream>

#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/petsc/TaoReducedFunctionalAdapter.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class QuadraticReducedFunctional final : public inverse::ReducedFunctional
{
public:
  index_type numParams() const override
  {
    return 2;
  }

  real_type value(const Vector& params) override
  {
    return 0.5 * (params[0] - 1.0) * (params[0] - 1.0)
           + (params[1] + 2.0) * (params[1] + 2.0);
  }

  void grad(const Vector& params, Vector& out) override
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

class TaoReducedFunctionalAdapterTests : public TestBase
{
public:
  TestOutcome callbackEvaluatesValueAndGradient()
  {
    TestStatus status;
    status = true;

    QuadraticReducedFunctional                  functional;
    inverse::TaoReducedFunctionalAdapter adapter(functional);

    Vec params   = nullptr;
    Vec gradient = nullptr;
    Tao tao      = nullptr;

    PetscErrorCode ierr  = VecCreateSeq(PETSC_COMM_SELF, 2, &params);
    status              *= (ierr == 0);
    ierr                 = VecDuplicate(params, &gradient);
    status              *= (ierr == 0);
    ierr                 = TaoCreate(PETSC_COMM_SELF, &tao);
    status              *= (ierr == 0);

    const PetscInt    indices[2]  = {0, 1};
    const PetscScalar values[2]   = {0.25, -0.5};
    ierr                          = VecSetValues(params, 2, indices, values, INSERT_VALUES);
    status                       *= (ierr == 0);
    ierr                          = VecAssemblyBegin(params);
    status                       *= (ierr == 0);
    ierr                          = VecAssemblyEnd(params);
    status                       *= (ierr == 0);

    PetscReal value = 0.0;
    ierr            = inverse::TaoReducedFunctionalAdapter::
        formObjectiveAndGradient(tao, params, &value, gradient, &adapter);
    status *= (ierr == 0);

    status *= isEqual(value,
                      0.5 * (values[0] - 1.0) * (values[0] - 1.0)
                          + (values[1] + 2.0) * (values[1] + 2.0));

    const PetscScalar* gradient_values  = nullptr;
    ierr                                = VecGetArrayRead(gradient, &gradient_values);
    status                             *= (ierr == 0);
    status                             *= isEqual(PetscRealPart(gradient_values[0]), values[0] - 1.0);
    status                             *= isEqual(PetscRealPart(gradient_values[1]),
                      2.0 * (values[1] + 2.0));
    ierr                                = VecRestoreArrayRead(gradient, &gradient_values);
    status                             *= (ierr == 0);

    if (tao != nullptr)
    {
      TaoDestroy(&tao);
    }
    if (gradient != nullptr)
    {
      VecDestroy(&gradient);
    }
    if (params != nullptr)
    {
      VecDestroy(&params);
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
