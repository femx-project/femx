#pragma once

#include <petsctao.h>

#include <functional>
#include <stdexcept>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>

namespace femx
{
namespace opt
{

using TaoNumParamsCallback = std::function<Index()>;
using TaoValueGradCallback =
    std::function<Real(const Vector<Real>&, Vector<Real>&)>;

/** @brief Adapter from PETSc TAO callbacks to a reduced functional. */
class TaoReducedFunctionalAdapter
{
public:
  TaoReducedFunctionalAdapter(TaoNumParamsCallback nprm,
                              TaoValueGradCallback value_grad)
    : nprm_(std::move(nprm)),
      value_grad_(std::move(value_grad))
  {
  }

  template <typename Functional,
            typename = decltype(std::declval<Functional&>().numParams()),
            typename = decltype(std::declval<Functional&>().valueGrad(
                std::declval<const Vector<Real>&>(),
                std::declval<Vector<Real>&>()))>
  explicit TaoReducedFunctionalAdapter(Functional& fn)
    : TaoReducedFunctionalAdapter(
          [&fn]()
          { return fn.numParams(); },
          [&fn](const Vector<Real>& prm, Vector<Real>& grad)
          { return fn.valueGrad(prm, grad); })
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
                                      void*      ctx)
  {
    (void) tao;

    auto* adapter = static_cast<TaoReducedFunctionalAdapter*>(ctx);
    if (adapter == nullptr || value == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }

    try
    {
      PetscCall(::femx::linalg::detail::copyFromPETSc(
          prm, adapter->prm_));
      if (adapter->prm_.size() != adapter->numParams())
      {
        throw std::runtime_error(
            "TAO parameter vector size does not match reduced functional");
      }

      *value = static_cast<PetscReal>(
          adapter->value_grad_(adapter->prm_, adapter->grad_));

      if (grad != nullptr)
      {
        PetscCall(::femx::linalg::detail::copyToPETSc(
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
  Index numParams() const
  {
    if (!nprm_)
    {
      throw std::runtime_error(
          "TAO reduced functional adapter has no size callback");
    }
    return nprm_();
  }

private:
  TaoNumParamsCallback nprm_;
  TaoValueGradCallback value_grad_;
  Vector<Real>         prm_;
  Vector<Real>         grad_;
};

} // namespace opt
} // namespace femx
