#pragma once

#include <stdexcept>
#include <utility>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/TimeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

using LocalTimeVolumeResidualFunction =
    void (*)(Index       step,
             Index       cell,
             Index       nq,
             Index       nn,
             Index       dim,
             Index       nres,
             Index       num_prev_states,
             Index       num_next_states,
             Index       nprm,
             const Real* N,
             const Real* dNdx,
             const Real* JxW,
             const Real* prev,
             const Real* nxt,
             const Real* prm,
             Real*       out);

/** @brief Enzyme time element kernel with volume element values. */
template <LocalTimeVolumeResidualFunction Residual>
class EnzymeTimeVolumeKernel final : public TimeElementKernel
{
public:
  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  nres,
                         Index                  num_prev_states,
                         Index                  num_next_states,
                         Index                  nprm)
    : space_(space),
      quad_(quadrature),
      nres_(nres),
      num_prev_states_(num_prev_states),
      num_next_states_(num_next_states),
      num_variable_prm_(nprm),
      nprm_(nprm)
  {
    checkDimensions();
  }

  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  nres,
                         Index                  num_prev_states,
                         Index                  num_next_states,
                         Index                  num_variable_prm,
                         Vector<Real>           fixed_prm)
    : space_(space),
      quad_(quadrature),
      nres_(nres),
      num_prev_states_(num_prev_states),
      num_next_states_(num_next_states),
      num_variable_prm_(num_variable_prm),
      nprm_(num_variable_prm + fixed_prm.size()),
      fixed_prm_(std::move(fixed_prm))
  {
    checkDimensions();
  }

  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  nres,
                         Index                  num_prev_states,
                         Index                  num_next_states,
                         Vector<Real>           fixed_prm)
    : EnzymeTimeVolumeKernel(space,
                             quadrature,
                             nres,
                             num_prev_states,
                             num_next_states,
                             0,
                             std::move(fixed_prm))
  {
  }

  void checkDimensions()
  {
    if (nres_ < 0 || num_prev_states_ < 0
        || num_next_states_ < 0 || num_variable_prm_ < 0 || nprm_ < 0)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel received invalid dimensions");
    }
    if (num_next_states_ <= 0 || num_prev_states_ % num_next_states_ != 0)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel history size must be a multiple of next-state size");
    }
    num_history_states_     = num_prev_states_ / num_next_states_;
    num_history_state_dofs_ = num_next_states_;
  }

  void res(Index                    step,
           Index                    ic,
           problem::TimeHistoryView hist,
           const Vector<Real>&      nxt,
           const Vector<Real>&      prm,
           Vector<Real>&            out) const override
  {
    checkInputSizes(hist, nxt, prm);
    resizeOrZero(out, nres_);
    const Vector<Real> residual_prm = makeResidualPrm(prm);

    ElementValues vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().cell(ic));
    Residual(step,
             ic,
             vals.numQuadraturePoints(),
             vals.numNodes(),
             vals.dim(),
             nres_,
             num_prev_states_,
             num_next_states_,
             nprm_,
             vals.NData(),
             vals.dNdxData(),
             vals.JxWData(),
             hist.data(),
             nxt.data(),
             residual_prm.data(),
             out.data());
  }

  void jacobian(Index                    step,
                Index                    ic,
                problem::VariableBlock   wrt,
                problem::TimeHistoryView hist,
                const Vector<Real>&      nxt,
                const Vector<Real>&      prm,
                DenseMatrix&             out) const override
  {
    checkInputSizes(hist, nxt, prm);

    if (wrt.isHistoryState())
    {
      historyStateJac(
          step, ic, wrt.historyLag(), hist, nxt, prm, out);
      return;
    }
    if (wrt.isNextState())
    {
      nextStateJac(step, ic, hist, nxt, prm, out);
      return;
    }
    paramJac(step, ic, hist, nxt, prm, out);
  }

private:
  void historyStateJac(Index                    step,
                       Index                    ic,
                       Index                    lag,
                       problem::TimeHistoryView hist,
                       const Vector<Real>&      nxt,
                       const Vector<Real>&      prm,
                       DenseMatrix&             out) const
  {
    if (lag < 0 || lag >= num_history_states_)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel history lag is out of range");
    }

    out.resize(nres_, num_history_state_dofs_);
    if (num_history_state_dofs_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(nres_);
    Vector<Real> out_adj(nres_);
    Vector<Real> previous_adj(num_prev_states_);

    for (Index row = 0; row < nres_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      previous_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              step,
                              enzyme_const,
                              ic,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              nres_,
                              enzyme_const,
                              num_prev_states_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              nprm_,
                              enzyme_const,
                              vals.NData(),
                              enzyme_const,
                              vals.dNdxData(),
                              enzyme_const,
                              vals.JxWData(),
                              enzyme_dup,
                              hist.data(),
                              previous_adj.data(),
                              enzyme_const,
                              nxt.data(),
                              enzyme_const,
                              residual_prm.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      const Index offset = lag * num_history_state_dofs_;
      for (Index col = 0; col < num_history_state_dofs_; ++col)
      {
        out(row, col) = previous_adj[offset + col];
      }
    }
#else
    (void) step;
    (void) ic;
    (void) hist;
    (void) nxt;
    (void) prm;
    throwUnavailable();
#endif
  }

  void nextStateJac(Index                    step,
                    Index                    ic,
                    problem::TimeHistoryView hist,
                    const Vector<Real>&      nxt,
                    const Vector<Real>&      prm,
                    DenseMatrix&             out) const
  {
    out.resize(nres_, num_next_states_);
    if (num_next_states_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(nres_);
    Vector<Real> out_adj(nres_);
    Vector<Real> next_adj(num_next_states_);

    for (Index row = 0; row < nres_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      next_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              step,
                              enzyme_const,
                              ic,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              nres_,
                              enzyme_const,
                              num_prev_states_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              nprm_,
                              enzyme_const,
                              vals.NData(),
                              enzyme_const,
                              vals.dNdxData(),
                              enzyme_const,
                              vals.JxWData(),
                              enzyme_const,
                              hist.data(),
                              enzyme_dup,
                              nxt.data(),
                              next_adj.data(),
                              enzyme_const,
                              residual_prm.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < num_next_states_; ++col)
      {
        out(row, col) = next_adj[col];
      }
    }
#else
    (void) step;
    (void) ic;
    (void) hist;
    (void) nxt;
    (void) prm;
    throwUnavailable();
#endif
  }

  void paramJac(Index                    step,
                Index                    ic,
                problem::TimeHistoryView hist,
                const Vector<Real>&      nxt,
                const Vector<Real>&      prm,
                DenseMatrix&             out) const
  {
    out.resize(nres_, num_variable_prm_);
    if (num_variable_prm_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(nres_);
    Vector<Real> out_adj(nres_);
    Vector<Real> param_adj(nprm_);

    for (Index row = 0; row < nres_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      param_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              step,
                              enzyme_const,
                              ic,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              nres_,
                              enzyme_const,
                              num_prev_states_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              nprm_,
                              enzyme_const,
                              vals.NData(),
                              enzyme_const,
                              vals.dNdxData(),
                              enzyme_const,
                              vals.JxWData(),
                              enzyme_const,
                              hist.data(),
                              enzyme_const,
                              nxt.data(),
                              enzyme_dup,
                              residual_prm.data(),
                              param_adj.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < num_variable_prm_; ++col)
      {
        out(row, col) = param_adj[col];
      }
    }
#else
    (void) step;
    (void) ic;
    (void) hist;
    (void) nxt;
    (void) prm;
    throwUnavailable();
#endif
  }

  void checkInputSizes(problem::TimeHistoryView hist,
                       const Vector<Real>&      nxt,
                       const Vector<Real>&      prm) const
  {
    if (hist.count() != num_history_states_
        || hist.stateSize() != num_history_state_dofs_
        || nxt.size() != num_next_states_
        || prm.size() != num_variable_prm_)
    {
      throw std::runtime_error("EnzymeTimeVolumeKernel input size mismatch");
    }
  }

  Vector<Real> makeResidualPrm(const Vector<Real>& variable_prm) const
  {
    if (fixed_prm_.empty())
    {
      return variable_prm;
    }

    Vector<Real> out(nprm_);
    for (Index i = 0; i < num_variable_prm_; ++i)
    {
      out[i] = variable_prm[i];
    }
    for (Index i = 0; i < fixed_prm_.size(); ++i)
    {
      out[num_variable_prm_ + i] = fixed_prm_[i];
    }
    return out;
  }

  [[noreturn]] static void throwUnavailable()
  {
    throw std::runtime_error(
        "EnzymeTimeVolumeKernel requires FEMX_ENABLE_ENZYME=ON");
  }

private:
  const FESpace&         space_;
  const GaussQuadrature& quad_;
  Index                  nres_{0};
  Index                  num_prev_states_{0};
  Index                  num_next_states_{0};
  Index                  num_history_states_{1};
  Index                  num_history_state_dofs_{0};
  Index                  num_variable_prm_{0};
  Index                  nprm_{0};
  Vector<Real>           fixed_prm_;
};

} // namespace assembly
} // namespace femx
