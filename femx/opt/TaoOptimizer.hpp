#pragma once

#include <petsctao.h>

#include <cmath>
#include <string>
#include <utility>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/petsc/PETScBackend.hpp>
#include <femx/opt/TaoReducedFunctionalAdapter.hpp>

namespace femx
{
namespace opt
{

/**
 * @brief PETSc/TAO algorithm and convergence options.
 */
struct TaoOptions
{
  std::string type = TAOLMVM; ///< PETSc/TAO solver type.

  Real abs_tol            = 1.0e-8; ///< Absolute gradient tolerance.
  Real rel_tol            = 1.0e-8; ///< Gradient-to-objective tolerance.
  Real grad_reduction_tol = 0.0;    ///< Gradient reduction tolerance.

  Index max_its = 100; ///< Maximum TAO iterations.
};

/**
 * @brief Result returned by TaoOptimizer::solve.
 */
struct TaoResult
{
  HostVector         prm;                                        ///< Final parameter vector.
  HostVector         grad;                                       ///< Final reduced gradient.
  Real               value             = 0.0;                    ///< Final objective value.
  Real               grad_norm_squared = 0.0;                    ///< Squared final gradient norm.
  Index              its               = 0;                      ///< Number of TAO iterations.
  TaoConvergedReason reason            = TAO_CONTINUE_ITERATING; ///< TAO reason.

  bool converged() const
  {
    return static_cast<int>(reason) > 0;
  }
};

/**
 * @brief Lower and upper bounds for bound-constrained optimization.
 */
struct TaoBounds
{
  HostVector lower; ///< Lower bound for each parameter.
  HostVector upper; ///< Upper bound for each parameter.
};

/**
 * @brief Per-iteration information passed to TaoProgressMonitor.
 */
struct TaoIterationInfo
{
  Index              its             = 0;                      ///< Current TAO iteration.
  Real               value           = 0.0;                    ///< Current objective value.
  Real               grad_norm       = 0.0;                    ///< Current gradient norm.
  Real               constraint_norm = 0.0;                    ///< Current constraint norm.
  Real               step_norm       = 0.0;                    ///< Current step norm.
  TaoConvergedReason reason          = TAO_CONTINUE_ITERATING; ///< TAO reason.
  HostVector         grad;                                     ///< Current reduced gradient.
};

/**
 * @brief Observer interface for PETSc/TAO progress reporting.
 *
 * Implementations receive iteration diagnostics and the current parameter
 * vector during optimization.
 */
class TaoProgressMonitor
{
public:
  virtual ~TaoProgressMonitor() = default;

  virtual void observe(const TaoIterationInfo& info,
                       const HostVector&       curr_prm) = 0;
};

/**
 * @brief PETSc/TAO optimizer for reduced functionals.
 *
 * TaoOptimizer adapts femx value-and-gradient callbacks for PETSc/TAO while
 * exposing vectors through the lightweight femx Vector container in its public API.
 */
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
  TaoOptimizer(TaoNumParamsCallback num_param,
               TaoValueGradCallback value_grad,
               MPI_Comm             comm = PETSC_COMM_SELF)
    : num_param_(std::move(num_param)),
      value_grad_(std::move(value_grad)),
      comm_(comm)
  {
  }

  template <typename Functional,
            typename = decltype(std::declval<Functional&>().numParams()),
            typename = decltype(std::declval<Functional&>().valueGrad(
                std::declval<const HostVector&>(),
                std::declval<HostVector&>()))>
  explicit TaoOptimizer(Functional& fn,
                        MPI_Comm    comm = PETSC_COMM_SELF)
    : TaoOptimizer(
          [&fn]()
          { return fn.numParams(); },
          [&fn](const HostVector& prm, HostVector& grad)
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

  void setBounds(const HostVector& lower, const HostVector& upper)
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

  void setVariableScale(const HostVector& scale)
  {
    scale_     = scale;
    has_scale_ = true;
  }

  void clearVariableScale()
  {
    scale_.clear();
    has_scale_ = false;
  }

  void setMonitor(TaoProgressMonitor* monitor)
  {
    progress_monitor_ = monitor;
  }

  void clearMonitor()
  {
    progress_monitor_ = nullptr;
  }

  PetscErrorCode solve(const HostVector& init, TaoResult& result)
  {
    if (!num_param_ || !value_grad_)
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
      const HostVector opt_init = toOptimizerParam(init);
      PetscCall(
          ::femx::linalg::detail::copyToPETSc(opt_init.view(), prm.get()));

      TaoReducedFunctionalAdapter adapter(
          num_param_,
          [this](const HostVector& opt_prm, HostVector& opt_grad)
          {
            return valueGradInOptimizerCoordinates(opt_prm, opt_grad);
          });

      PetscCall(TaoCreate(comm_, tao.put()));
      PetscCall(TaoSetType(tao.get(), taoType()));
      PetscCall(TaoSetSolution(tao.get(), prm.get()));
      PetscCall(adapter.setValueGrad(tao.get()));
      if (progress_monitor_ != nullptr)
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
        const HostVector opt_lower = toOptimizerParam(bounds_.lower);
        const HostVector opt_upper = toOptimizerParam(bounds_.upper);
        PetscCall(::femx::linalg::detail::copyToPETSc(opt_lower.view(),
                                                      lower.get()));
        PetscCall(::femx::linalg::detail::copyToPETSc(opt_upper.view(),
                                                      upper.get()));
        PetscCall(TaoSetVariableBounds(tao.get(), lower.get(), upper.get()));
      }
      PetscCall(TaoSetTolerances(
          tao.get(),
          static_cast<PetscReal>(opts_.abs_tol),
          static_cast<PetscReal>(opts_.rel_tol),
          static_cast<PetscReal>(opts_.grad_reduction_tol)));
      PetscCall(TaoSetMaximumIterations(tao.get(), static_cast<PetscInt>(opts_.max_its)));
      PetscCall(TaoSetFromOptions(tao.get()));
      PetscCall(TaoSolve(tao.get()));

      HostVector opt_result;
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
    if (self == nullptr || self->progress_monitor_ == nullptr)
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

    HostVector opt_curr;
    PetscCall(::femx::linalg::detail::copyFromPETSc(prm, opt_curr));
    HostVector curr = self->toPhysicalParam(opt_curr);

    Vec grad = nullptr;
    PetscCall(TaoGetGradient(tao, &grad, nullptr, nullptr));
    HostVector curr_grad;
    if (grad != nullptr)
    {
      HostVector opt_grad;
      PetscCall(::femx::linalg::detail::copyFromPETSc(grad, opt_grad));
      curr_grad = self->toPhysicalGrad(opt_grad);
      grad_norm =
          static_cast<PetscReal>(std::sqrt(squaredNorm(curr_grad)));
    }

    TaoIterationInfo info{
        static_cast<Index>(its),
        static_cast<Real>(value),
        static_cast<Real>(grad_norm),
        static_cast<Real>(constraint_norm),
        static_cast<Real>(step_norm),
        reason,
        std::move(curr_grad)};

    try
    {
      self->progress_monitor_->observe(info, curr);
    }
    catch (...)
    {
      return PETSC_ERR_LIB;
    }

    return PETSC_SUCCESS;
  }

  Index numParams() const
  {
    return num_param_();
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

  Real valueGradInOptimizerCoordinates(const HostVector& opt_prm,
                                       HostVector&       opt_grad)
  {
    const HostVector prm = toPhysicalParam(opt_prm);
    HostVector       grad;
    const Real       value = value_grad_(prm, grad);
    opt_grad               = toOptimizerGrad(grad);
    return value;
  }

  HostVector toOptimizerParam(const HostVector& prm) const
  {
    if (!has_scale_)
    {
      return prm;
    }
    HostVector out(prm.size());
    for (Index i = 0; i < prm.size(); ++i)
    {
      out[i] = prm[i] / scale_[i];
    }
    return out;
  }

  HostVector toPhysicalParam(const HostVector& opt_prm) const
  {
    if (!has_scale_)
    {
      return opt_prm;
    }
    HostVector out(opt_prm.size());
    for (Index i = 0; i < opt_prm.size(); ++i)
    {
      out[i] = scale_[i] * opt_prm[i];
    }
    return out;
  }

  HostVector toOptimizerGrad(const HostVector& grad) const
  {
    if (!has_scale_)
    {
      return grad;
    }
    HostVector out(grad.size());
    for (Index i = 0; i < grad.size(); ++i)
    {
      out[i] = scale_[i] * grad[i];
    }
    return out;
  }

  HostVector toPhysicalGrad(const HostVector& opt_grad) const
  {
    if (!has_scale_)
    {
      return opt_grad;
    }
    HostVector out(opt_grad.size());
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
    const PetscInt num_local_dofs = comm_size == 1 ? size : PETSC_DECIDE;

    PetscCall(VecCreate(comm, vec.put()));
    PetscCall(VecSetSizes(vec.get(), num_local_dofs, size));
    PetscCall(VecSetFromOptions(vec.get()));
    return PETSC_SUCCESS;
  }

private:
  TaoNumParamsCallback num_param_;
  TaoValueGradCallback value_grad_;
  MPI_Comm             comm_{PETSC_COMM_SELF};
  TaoOptions           opts_;
  TaoBounds            bounds_;
  bool                 has_bounds_{false};
  HostVector           scale_;
  bool                 has_scale_{false};
  TaoProgressMonitor*  progress_monitor_{nullptr};
};

} // namespace opt
} // namespace femx
