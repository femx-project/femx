#pragma once

#include <petsctao.h>

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/petsc/VectorConversion.hpp>

namespace femx
{
namespace inverse
{

/** @brief Adapter from PETSc TAO callbacks to a ReducedFunctional. */
class TaoReducedFunctionalAdapter
{
public:
  explicit TaoReducedFunctionalAdapter(ReducedFunctional& functional)
    : functional_(&functional)
  {
  }

  PetscErrorCode setValueGrad(Tao tao, Vec grad_template = nullptr)
  {
    return TaoSetObjectiveAndGradient(
        tao, grad_template, formValueGrad, this);
  }

  static PetscErrorCode formValueGrad(Tao        tao,
                                      Vec        prm,
                                      PetscReal* value,
                                      Vec        grad,
                                      void*      context)
  {
    (void) tao;

    auto* adapter = static_cast<TaoReducedFunctionalAdapter*>(context);
    if (adapter == nullptr || value == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(::femx::system::detail::copyFromPETSc(
          prm, adapter->prm_));
      if (adapter->prm_.size() != adapter->functional_->numParams())
      {
        throw std::runtime_error(
            "TAO parameter vector size does not match ReducedFunctional");
      }

      *value = static_cast<PetscReal>(
          adapter->functional_->valueGrad(adapter->prm_,
                                          adapter->grad_));

      if (grad != nullptr)
      {
        PetscCall(::femx::system::detail::copyToPETSc(
            adapter->grad_, grad));
      }
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

private:
  ReducedFunctional* functional_{nullptr};
  Vector<Real>       prm_;
  Vector<Real>       grad_;
};

} // namespace inverse
} // namespace femx
