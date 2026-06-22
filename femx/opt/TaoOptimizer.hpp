#pragma once

#include <petsctao.h>

#include <cmath>
#include <functional>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/VectorConversion.hpp>
#include <femx/opt/TaoReducedFunctionalAdapter.hpp>

namespace femx
{
namespace opt
{

struct TaoOptions
{
  std::string type = TAOLMVM;

  Real abs_tol  = 1.0e-8;
  Real rel_tol  = 1.0e-8;
  Real step_tol = 0.0;

  Index max_its = 100;
};

struct TaoResult
{
  Vector<Real>       prm;
  Vector<Real>       grad;
  Real               value             = 0.0;
  Real               grad_norm_squared = 0.0;
  Index              its               = 0;
  TaoConvergedReason reason            = TAO_CONTINUE_ITERATING;

  bool converged() const
  {
    return static_cast<int>(reason) > 0;
  }
};

struct TaoBounds
{
  Vector<Real> lower;
  Vector<Real> upper;
};

struct TaoIterationInfo
{
  Index              its             = 0;
  Real               value           = 0.0;
  Real               grad_norm       = 0.0;
  Real               constraint_norm = 0.0;
  Real               step_norm       = 0.0;
  TaoConvergedReason reason          = TAO_CONTINUE_ITERATING;
  Vector<Real>       grad;
};

using TaoMonitorCallback =
    std::function<void(const TaoIterationInfo&, const Vector<Real>&)>;

/** @brief PETSc/TAO optimizer for reduced functionals. */
class TaoOptimizer
{
private:
  class ScopedVec
  {
  public:
    ~ScopedVec()
    {
      if (vec_ != nullptr)
      {
        VecDestroy(&vec_);
      }
    }

    Vec get() const
    {
      return vec_;
    }

    Vec* put()
    {
      return &vec_;
    }

  private:
    Vec vec_{nullptr};
  };

  class ScopedTao
  {
  public:
    ~ScopedTao()
    {
      if (tao_ != nullptr)
      {
        TaoDestroy(&tao_);
      }
    }

    Tao get() const
    {
      return tao_;
    }

    Tao* put()
    {
      return &tao_;
    }

  private:
    Tao tao_{nullptr};
  };

public:
  TaoOptimizer(TaoNumParamsCallback nprm,
               TaoValueGradCallback value_grad,
               MPI_Comm             comm = PETSC_COMM_SELF)
    : nprm_(std::move(nprm)),
      value_grad_(std::move(value_grad)),
      comm_(comm)
  {
  }

  template <typename Functional,
            typename = decltype(std::declval<Functional&>().numParams()),
            typename = decltype(std::declval<Functional&>().valueGrad(
                std::declval<const Vector<Real>&>(),
                std::declval<Vector<Real>&>()))>
  explicit TaoOptimizer(Functional& fn,
                        MPI_Comm    comm = PETSC_COMM_SELF)
    : TaoOptimizer(
          [&fn]()
          { return fn.numParams(); },
          [&fn](const Vector<Real>& prm, Vector<Real>& grad)
          { return fn.valueGrad(prm, grad); },
          comm)
  {
  }

  TaoOptions& opts()
  {
    return opts_;
  }

  const TaoOptions& opts() const
  {
    return opts_;
  }

  void setBounds(const Vector<Real>& lower, const Vector<Real>& upper)
  {
    bounds_.lower = lower;
    bounds_.upper = upper;
    has_bounds_   = true;
  }

  void clearBounds()
  {
    bounds_     = TaoBounds{};
    has_bounds_ = false;
  }

  bool hasBounds() const
  {
    return has_bounds_;
  }

  const TaoBounds& bnds() const
  {
    return bounds_;
  }

  void setVariableScale(const Vector<Real>& scale)
  {
    scale_     = scale;
    has_scale_ = true;
  }

  void clearVariableScale()
  {
    scale_.clear();
    has_scale_ = false;
  }

  void setMonitor(TaoMonitorCallback monitor)
  {
    monitor_ = std::move(monitor);
  }

  void clearMonitor()
  {
    monitor_ = nullptr;
  }

  PetscErrorCode solve(const Vector<Real>& init, TaoResult& result)
  {
    if (!nprm_ || !value_grad_)
    {
      return PETSC_ERR_ARG_NULL;
    }
    if (init.size() != numParams())
    {
      return PETSC_ERR_ARG_SIZ;
    }
    const PetscErrorCode bounds_ierr = validateBounds();
    if (bounds_ierr != PETSC_SUCCESS)
    {
      return bounds_ierr;
    }
    const PetscErrorCode scale_ierr = validateScale();
    if (scale_ierr != PETSC_SUCCESS)
    {
      return scale_ierr;
    }

    PetscBool initialized = PETSC_FALSE;
    PetscCall(PetscInitialized(&initialized));
    if (initialized != PETSC_TRUE)
    {
      return PETSC_ERR_ORDER;
    }

    ScopedVec prm;
    ScopedVec lower;
    ScopedVec upper;
    ScopedTao tao;

    try
    {
      PetscCall(createVec(comm_, static_cast<PetscInt>(numParams()), prm));
      const Vector<Real> opt_init = toOptimizerParam(init);
      PetscCall(::femx::linalg::detail::copyToPETSc(opt_init, prm.get()));

      TaoReducedFunctionalAdapter adapter(
          nprm_,
          [this](const Vector<Real>& opt_prm, Vector<Real>& opt_grad)
          {
            return valueGradInOptimizerCoordinates(opt_prm, opt_grad);
          });

      PetscCall(TaoCreate(comm_, tao.put()));
      PetscCall(TaoSetType(tao.get(), taoType()));
      PetscCall(TaoSetSolution(tao.get(), prm.get()));
      PetscCall(adapter.setValueGrad(tao.get()));
      if (monitor_)
      {
#if PETSC_VERSION_GE(3, 21, 0)
        PetscCall(TaoMonitorSet(
            tao.get(), &TaoOptimizer::monitorCallback, this, nullptr));
#else
        PetscCall(TaoSetMonitor(
            tao.get(), &TaoOptimizer::monitorCallback, this, nullptr));
#endif
      }
      if (has_bounds_)
      {
        PetscCall(VecDuplicate(prm.get(), lower.put()));
        PetscCall(VecDuplicate(prm.get(), upper.put()));
        const Vector<Real> opt_lower = toOptimizerParam(bounds_.lower);
        const Vector<Real> opt_upper = toOptimizerParam(bounds_.upper);
        PetscCall(::femx::linalg::detail::copyToPETSc(opt_lower, lower.get()));
        PetscCall(::femx::linalg::detail::copyToPETSc(opt_upper, upper.get()));
        PetscCall(TaoSetVariableBounds(tao.get(), lower.get(), upper.get()));
      }
      PetscCall(TaoSetTolerances(
          tao.get(),
          static_cast<PetscReal>(opts_.abs_tol),
          static_cast<PetscReal>(opts_.rel_tol),
          static_cast<PetscReal>(opts_.step_tol)));
      PetscCall(TaoSetMaximumIterations(
          tao.get(), static_cast<PetscInt>(opts_.max_its)));
      PetscCall(TaoSetFromOptions(tao.get()));
      PetscCall(TaoSolve(tao.get()));

      Vector<Real> opt_result;
      PetscCall(::femx::linalg::detail::copyFromPETSc(prm.get(), opt_result));
      result.prm               = toPhysicalParam(opt_result);
      result.value             = value_grad_(result.prm, result.grad);
      result.grad_norm_squared = squaredNorm(result.grad);

      PetscInt its = 0;
      PetscCall(TaoGetIterationNumber(tao.get(), &its));
      result.its = static_cast<Index>(its);

      PetscCall(TaoGetConvergedReason(tao.get(), &result.reason));
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

private:
  static PetscErrorCode monitorCallback(Tao tao, void* ctx)
  {
    auto* self = static_cast<TaoOptimizer*>(ctx);
    if (self == nullptr || !self->monitor_)
    {
      return PETSC_SUCCESS;
    }

    PetscInt           its   = 0;
    PetscReal          value = 0.0, grad_norm = 0.0, constraint_norm = 0.0;
    PetscReal          step_norm = 0.0;
    TaoConvergedReason reason    = TAO_CONTINUE_ITERATING;
    PetscCall(TaoGetSolutionStatus(tao,
                                   &its,
                                   &value,
                                   &grad_norm,
                                   &constraint_norm,
                                   &step_norm,
                                   &reason));

    Vec prm = nullptr;
    PetscCall(TaoGetSolution(tao, &prm));

    Vector<Real> opt_current;
    PetscCall(::femx::linalg::detail::copyFromPETSc(prm, opt_current));
    Vector<Real> current = self->toPhysicalParam(opt_current);

    Vec grad = nullptr;
    PetscCall(TaoGetGradient(tao, &grad, nullptr, nullptr));
    Vector<Real> current_grad;
    if (grad != nullptr)
    {
      Vector<Real> opt_grad;
      PetscCall(::femx::linalg::detail::copyFromPETSc(grad, opt_grad));
      current_grad = self->toPhysicalGrad(opt_grad);
      grad_norm =
          static_cast<PetscReal>(std::sqrt(squaredNorm(current_grad)));
    }

    TaoIterationInfo info{
        static_cast<Index>(its),
        static_cast<Real>(value),
        static_cast<Real>(grad_norm),
        static_cast<Real>(constraint_norm),
        static_cast<Real>(step_norm),
        reason,
        std::move(current_grad)};

    try
    {
      self->monitor_(info, current);
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

  Index numParams() const
  {
    return nprm_();
  }

  const char* taoType() const
  {
    if (has_bounds_ && opts_.type == TAOLMVM)
    {
      return TAOBLMVM;
    }
    return opts_.type.c_str();
  }

  PetscErrorCode validateBounds() const
  {
    if (!has_bounds_)
    {
      return PETSC_SUCCESS;
    }
    if (bounds_.lower.size() != numParams()
        || bounds_.upper.size() != numParams())
    {
      return PETSC_ERR_ARG_SIZ;
    }
    for (Index i = 0; i < numParams(); ++i)
    {
      if (bounds_.lower[i] > bounds_.upper[i])
      {
        return PETSC_ERR_ARG_OUTOFRANGE;
      }
    }
    return PETSC_SUCCESS;
  }

  PetscErrorCode validateScale() const
  {
    if (!has_scale_)
    {
      return PETSC_SUCCESS;
    }
    if (scale_.size() != numParams())
    {
      return PETSC_ERR_ARG_SIZ;
    }
    for (Index i = 0; i < scale_.size(); ++i)
    {
      if (!std::isfinite(scale_[i]) || scale_[i] <= 0.0)
      {
        return PETSC_ERR_ARG_OUTOFRANGE;
      }
    }
    return PETSC_SUCCESS;
  }

  Real valueGradInOptimizerCoordinates(const Vector<Real>& opt_prm,
                                       Vector<Real>&       opt_grad)
  {
    const Vector<Real> prm = toPhysicalParam(opt_prm);
    Vector<Real>       grad;
    const Real         value = value_grad_(prm, grad);
    opt_grad                 = toOptimizerGrad(grad);
    return value;
  }

  Vector<Real> toOptimizerParam(const Vector<Real>& prm) const
  {
    if (!has_scale_)
    {
      return prm;
    }
    Vector<Real> out(prm.size());
    for (Index i = 0; i < prm.size(); ++i)
    {
      out[i] = prm[i] / scale_[i];
    }
    return out;
  }

  Vector<Real> toPhysicalParam(const Vector<Real>& opt_prm) const
  {
    if (!has_scale_)
    {
      return opt_prm;
    }
    Vector<Real> out(opt_prm.size());
    for (Index i = 0; i < opt_prm.size(); ++i)
    {
      out[i] = scale_[i] * opt_prm[i];
    }
    return out;
  }

  Vector<Real> toOptimizerGrad(const Vector<Real>& grad) const
  {
    if (!has_scale_)
    {
      return grad;
    }
    Vector<Real> out(grad.size());
    for (Index i = 0; i < grad.size(); ++i)
    {
      out[i] = scale_[i] * grad[i];
    }
    return out;
  }

  Vector<Real> toPhysicalGrad(const Vector<Real>& opt_grad) const
  {
    if (!has_scale_)
    {
      return opt_grad;
    }
    Vector<Real> out(opt_grad.size());
    for (Index i = 0; i < opt_grad.size(); ++i)
    {
      out[i] = opt_grad[i] / scale_[i];
    }
    return out;
  }

  static PetscErrorCode createVec(MPI_Comm   comm,
                                  PetscInt   size,
                                  ScopedVec& vec)
  {
    PetscMPIInt comm_size = 1;
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));
    const PetscInt nloc = comm_size == 1 ? size : PETSC_DECIDE;

    PetscCall(VecCreate(comm, vec.put()));
    PetscCall(VecSetSizes(vec.get(), nloc, size));
    PetscCall(VecSetFromOptions(vec.get()));
    return PETSC_SUCCESS;
  }

  static Real squaredNorm(const Vector<Real>& x)
  {
    Real value = 0.0;
    for (Index i = 0; i < x.size(); ++i)
    {
      value += x[i] * x[i];
    }
    return value;
  }

private:
  TaoNumParamsCallback nprm_;
  TaoValueGradCallback value_grad_;
  MPI_Comm             comm_{PETSC_COMM_SELF};
  TaoOptions           opts_;
  TaoBounds            bounds_;
  bool                 has_bounds_{false};
  Vector<Real>         scale_;
  bool                 has_scale_{false};
  TaoMonitorCallback   monitor_;
};

} // namespace opt
} // namespace femx
