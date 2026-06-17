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

  PetscErrorCode setObjectiveAndGradient(Tao tao, Vec gradient_template = nullptr)
  {
    return TaoSetObjectiveAndGradient(
        tao, gradient_template, formObjectiveAndGradient, this);
  }

  static PetscErrorCode formObjectiveAndGradient(Tao        tao,
                                                 Vec        params,
                                                 PetscReal* value,
                                                 Vec        gradient,
                                                 void*      context)
  {
    (void) tao;

    auto* adapter =
        static_cast<TaoReducedFunctionalAdapter*>(context);
    if (adapter == nullptr || value == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(::femx::system::detail::copyFromPETSc(
          params, adapter->params_));
      if (adapter->params_.size() != adapter->functional_->numParams())
      {
        throw std::runtime_error(
            "TAO parameter vector size does not match ReducedFunctional");
      }

      *value = static_cast<PetscReal>(
          adapter->functional_->valueGrad(adapter->params_,
                                          adapter->gradient_));

      if (gradient != nullptr)
      {
        PetscCall(::femx::system::detail::copyToPETSc(
            adapter->gradient_, gradient));
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
  Vector<Real>       params_;
  Vector<Real>       gradient_;
};

} // namespace inverse
} // namespace femx
