#pragma once

#include <petsctao.h>

#include <functional>
#include <string>
#include <utility>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/petsc/TaoReducedFunctionalAdapter.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/petsc/VectorConversion.hpp>

namespace femx
{
namespace inverse
{

struct TaoOptions
{
  std::string type = TAOLMVM;

  Real grad_abs_tolerance  = 1.0e-8;
  Real grad_rel_tolerance  = 1.0e-8;
  Real grad_step_tolerance = 0.0;

  Index max_its     = 100;
  bool  use_opts_db = true;
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
};

using TaoMonitorCallback =
    std::function<void(const TaoIterationInfo&, const Vector<Real>&)>;

/** @brief PETSc/TAO optimizer for a ReducedFunctional. */
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
  explicit TaoOptimizer(ReducedFunctional& functional,
                        MPI_Comm           comm = PETSC_COMM_SELF)
    : functional_(&functional),
      comm_(comm)
  {
  }

  TaoOptions& options()
  {
    return options_;
  }

  const TaoOptions& options() const
  {
    return options_;
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

  const TaoBounds& bounds() const
  {
    return bounds_;
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
    if (functional_ == nullptr)
    {
      return PETSC_ERR_ARG_NULL;
    }
    if (init.size() != functional_->numParams())
    {
      return PETSC_ERR_ARG_SIZ;
    }
    const PetscErrorCode bounds_ierr = validateBounds();
    if (bounds_ierr != PETSC_SUCCESS)
    {
      return bounds_ierr;
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
      PetscCall(createVec(comm_,
                          static_cast<PetscInt>(functional_->numParams()),
                          prm));
      PetscCall(::femx::system::detail::copyToPETSc(init,
                                                    prm.get()));

      TaoReducedFunctionalAdapter adapter(*functional_);

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
        PetscCall(::femx::system::detail::copyToPETSc(bounds_.lower,
                                                      lower.get()));
        PetscCall(::femx::system::detail::copyToPETSc(bounds_.upper,
                                                      upper.get()));
        PetscCall(TaoSetVariableBounds(tao.get(), lower.get(), upper.get()));
      }
      PetscCall(TaoSetTolerances(
          tao.get(),
          static_cast<PetscReal>(options_.grad_abs_tolerance),
          static_cast<PetscReal>(options_.grad_rel_tolerance),
          static_cast<PetscReal>(options_.grad_step_tolerance)));
      PetscCall(TaoSetMaximumIterations(
          tao.get(), static_cast<PetscInt>(options_.max_its)));
      if (options_.use_opts_db)
      {
        PetscCall(TaoSetFromOptions(tao.get()));
      }
      PetscCall(TaoSolve(tao.get()));

      PetscCall(::femx::system::detail::copyFromPETSc(prm.get(), result.prm));
      result.value             = functional_->valueGrad(result.prm, result.grad);
      result.grad_norm_squared = ::femx::squaredNorm(result.grad);

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

    Vector<Real> current;
    PetscCall(::femx::system::detail::copyFromPETSc(prm, current));

    const TaoIterationInfo info{
        static_cast<Index>(its),
        static_cast<Real>(value),
        static_cast<Real>(grad_norm),
        static_cast<Real>(constraint_norm),
        static_cast<Real>(step_norm),
        reason};

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

  const char* taoType() const
  {
    if (has_bounds_ && options_.type == TAOLMVM)
    {
      return TAOBLMVM;
    }
    return options_.type.c_str();
  }

  PetscErrorCode validateBounds() const
  {
    if (!has_bounds_)
    {
      return PETSC_SUCCESS;
    }
    if (bounds_.lower.size() != functional_->numParams()
        || bounds_.upper.size() != functional_->numParams())
    {
      return PETSC_ERR_ARG_SIZ;
    }
    for (Index i = 0; i < functional_->numParams(); ++i)
    {
      if (bounds_.lower[i] > bounds_.upper[i])
      {
        return PETSC_ERR_ARG_OUTOFRANGE;
      }
    }
    return PETSC_SUCCESS;
  }

  static PetscErrorCode createVec(MPI_Comm comm,
                                  PetscInt size,
                                  ScopedVec& vec)
  {
    PetscMPIInt comm_size = 1;
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));
    const PetscInt local_size = comm_size == 1 ? size : PETSC_DECIDE;

    PetscCall(VecCreate(comm, vec.put()));
    PetscCall(VecSetSizes(vec.get(), local_size, size));
    PetscCall(VecSetFromOptions(vec.get()));
    return PETSC_SUCCESS;
  }

private:
  ReducedFunctional* functional_{nullptr};
  MPI_Comm           comm_{PETSC_COMM_SELF};
  TaoOptions         options_;
  TaoBounds          bounds_;
  bool               has_bounds_{false};
  TaoMonitorCallback monitor_;
};

} // namespace inverse
} // namespace femx
