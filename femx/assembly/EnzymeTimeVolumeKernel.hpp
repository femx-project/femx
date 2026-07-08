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
             Index       elem,
             Index       num_qpts,
             Index       num_nodes,
             Index       dim,
             Index       num_residuals,
             Index       num_history_dofs,
             Index       num_next_states,
             Index       num_params,
             const Real* N,
             const Real* dNdx,
             const Real* JxW,
             const Real* hist,
             const Real* nxt,
             const Real* prm,
             Real*       out);

/**
 * @brief Enzyme time element kernel with volume element values.
 *
 * The template parameter is a local time-residual callback.  The kernel
 * evaluates the residual directly and uses Enzyme to form history, next-state,
 * and parameter Jacobian blocks when automatic differentiation is enabled.
 */
template <LocalTimeVolumeResidualFunction Residual>
class EnzymeTimeVolumeKernel final : public TimeElementKernel
{
public:
  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  num_residuals,
                         Index                  num_history_dofs,
                         Index                  num_next_states,
                         Index                  num_params)
    : space_(space),
      quad_(quadrature),
      num_residuals_(num_residuals),
      num_hist_dofs_(num_history_dofs),
      num_next_states_(num_next_states),
      num_variable_params_(num_params),
      num_params_(num_params)
  {
    checkDimensions();
  }

  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  num_residuals,
                         Index                  num_history_dofs,
                         Index                  num_next_states,
                         Index                  num_variable_params,
                         Vector<Real>           fixed_prm)
    : space_(space),
      quad_(quadrature),
      num_residuals_(num_residuals),
      num_hist_dofs_(num_history_dofs),
      num_next_states_(num_next_states),
      num_variable_params_(num_variable_params),
      num_params_(num_variable_params + fixed_prm.size()),
      fixed_prm_(std::move(fixed_prm))
  {
    checkDimensions();
  }

  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  num_residuals,
                         Index                  num_history_dofs,
                         Index                  num_next_states,
                         Vector<Real>           fixed_prm)
    : EnzymeTimeVolumeKernel(space,
                             quadrature,
                             num_residuals,
                             num_history_dofs,
                             num_next_states,
                             0,
                             std::move(fixed_prm))
  {
  }

  void checkDimensions()
  {
    if (num_residuals_ < 0 || num_hist_dofs_ < 0
        || num_next_states_ < 0 || num_variable_params_ < 0 || num_params_ < 0)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel received invalid dimensions");
    }
    if (num_next_states_ <= 0 || num_hist_dofs_ % num_next_states_ != 0)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel history size must be a multiple of next-state size");
    }
    num_hist_states_     = num_hist_dofs_ / num_next_states_;
    num_hist_state_dofs_ = num_next_states_;
  }

  void res(Index                    step,
           Index                    ie,
           problem::TimeHistoryView hist,
           const Vector<Real>&      nxt,
           const Vector<Real>&      prm,
           Vector<Real>&            out) const override
  {
    checkInputSizes(hist, nxt, prm);
    resizeOrZero(out, num_residuals_);
    const Vector<Real> residual_prm = makeResidualPrm(prm);

    ElementValues vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().elem(ie));
    Residual(step,
             ie,
             vals.numQuadraturePoints(),
             vals.numNodes(),
             vals.dim(),
             num_residuals_,
             num_hist_dofs_,
             num_next_states_,
             num_params_,
             vals.NData(),
             vals.dNdxData(),
             vals.JxWData(),
             hist.data(),
             nxt.data(),
             residual_prm.data(),
             out.data());
  }

  void jacobian(Index                    step,
                Index                    ie,
                problem::VariableBlock   wrt,
                problem::TimeHistoryView hist,
                const Vector<Real>&      nxt,
                const Vector<Real>&      prm,
                DenseMatrix&             out) const override
  {
    checkInputSizes(hist, nxt, prm);

    if (wrt.isHistoryState())
    {
      historyStateJac(step, ie, wrt.historyLag(), hist, nxt, prm, out);
      return;
    }
    if (wrt.isNextState())
    {
      nextStateJac(step, ie, hist, nxt, prm, out);
      return;
    }
    paramJac(step, ie, hist, nxt, prm, out);
  }

private:
  void historyStateJac(Index                    step,
                       Index                    ie,
                       Index                    lag,
                       problem::TimeHistoryView hist,
                       const Vector<Real>&      nxt,
                       const Vector<Real>&      prm,
                       DenseMatrix&             out) const
  {
    if (lag < 0 || lag >= num_hist_states_)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel history lag is out of range");
    }

    out.resize(num_residuals_, num_hist_state_dofs_);
    if (num_hist_state_dofs_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().elem(ie));

    Vector<Real> primal_out(num_residuals_);
    Vector<Real> out_adj(num_residuals_);
    Vector<Real> history_adj(num_hist_dofs_);

    for (Index row = 0; row < num_residuals_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      history_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              step,
                              enzyme_const,
                              ie,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              num_residuals_,
                              enzyme_const,
                              num_hist_dofs_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              num_params_,
                              enzyme_const,
                              vals.NData(),
                              enzyme_const,
                              vals.dNdxData(),
                              enzyme_const,
                              vals.JxWData(),
                              enzyme_dup,
                              hist.data(),
                              history_adj.data(),
                              enzyme_const,
                              nxt.data(),
                              enzyme_const,
                              residual_prm.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      const Index offset = lag * num_hist_state_dofs_;
      for (Index col = 0; col < num_hist_state_dofs_; ++col)
      {
        out(row, col) = history_adj[offset + col];
      }
    }
#else
    (void) step;
    (void) ie;
    (void) hist;
    (void) nxt;
    (void) prm;
    throwUnavailable();
#endif
  }

  void nextStateJac(Index                    step,
                    Index                    ie,
                    problem::TimeHistoryView hist,
                    const Vector<Real>&      nxt,
                    const Vector<Real>&      prm,
                    DenseMatrix&             out) const
  {
    out.resize(num_residuals_, num_next_states_);
    if (num_next_states_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().elem(ie));

    Vector<Real> primal_out(num_residuals_);
    Vector<Real> out_adj(num_residuals_);
    Vector<Real> next_adj(num_next_states_);

    for (Index row = 0; row < num_residuals_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      next_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              step,
                              enzyme_const,
                              ie,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              num_residuals_,
                              enzyme_const,
                              num_hist_dofs_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              num_params_,
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
    (void) ie;
    (void) hist;
    (void) nxt;
    (void) prm;
    throwUnavailable();
#endif
  }

  void paramJac(Index                    step,
                Index                    ie,
                problem::TimeHistoryView hist,
                const Vector<Real>&      nxt,
                const Vector<Real>&      prm,
                DenseMatrix&             out) const
  {
    out.resize(num_residuals_, num_variable_params_);
    if (num_variable_params_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      vals(space_.finiteElement(), quad_);
    vals.reinit(space_.mesh().elem(ie));

    Vector<Real> primal_out(num_residuals_);
    Vector<Real> out_adj(num_residuals_);
    Vector<Real> param_adj(num_params_);

    for (Index row = 0; row < num_residuals_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      param_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              step,
                              enzyme_const,
                              ie,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              num_residuals_,
                              enzyme_const,
                              num_hist_dofs_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              num_params_,
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

      for (Index col = 0; col < num_variable_params_; ++col)
      {
        out(row, col) = param_adj[col];
      }
    }
#else
    (void) step;
    (void) ie;
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
    if (hist.count() != num_hist_states_
        || hist.stateSize() != num_hist_state_dofs_
        || nxt.size() != num_next_states_
        || prm.size() != num_variable_params_)
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

    Vector<Real> out(num_params_);
    for (Index i = 0; i < num_variable_params_; ++i)
    {
      out[i] = variable_prm[i];
    }
    for (Index i = 0; i < fixed_prm_.size(); ++i)
    {
      out[num_variable_params_ + i] = fixed_prm_[i];
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
  Index                  num_residuals_{0};
  Index                  num_hist_dofs_{0};
  Index                  num_next_states_{0};
  Index                  num_hist_states_{1};
  Index                  num_hist_state_dofs_{0};
  Index                  num_variable_params_{0};
  Index                  num_params_{0};
  Vector<Real>           fixed_prm_;
};

} // namespace assembly
} // namespace femx
