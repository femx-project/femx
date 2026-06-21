#pragma once

#include <stdexcept>
#include <utility>

#include <femx/ad/Enzyme.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/assembly/TimeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>

namespace femx
{
namespace assembly
{

using LocalTimeVolumeResidualFunction =
    void (*)(Index       step,
             Index       cell,
             Index       num_qp,
             Index       num_nodes,
             Index       dim,
             Index       num_res,
             Index       num_prev_states,
             Index       num_next_states,
             Index       num_prm,
             const Real* N,
             const Real* dNdx,
             const Real* JxW,
             const Real* prev_state,
             const Real* next_state,
             const Real* prm,
             Real*       out);

/** @brief Enzyme time element kernel with volume element values. */
template <LocalTimeVolumeResidualFunction Residual>
class EnzymeTimeVolumeKernel final : public TimeElementKernel
{
public:
  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  num_res,
                         Index                  num_prev_states,
                         Index                  num_next_states,
                         Index                  num_prm)
    : space_(space),
      quad_(quadrature),
      num_res_(num_res),
      num_prev_states_(num_prev_states),
      num_next_states_(num_next_states),
      num_variable_prm_(num_prm),
      num_prm_(num_prm)
  {
    checkDimensions();
  }

  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  num_res,
                         Index                  num_prev_states,
                         Index                  num_next_states,
                         Index                  num_variable_prm,
                         Vector<Real>           fixed_prm)
    : space_(space),
      quad_(quadrature),
      num_res_(num_res),
      num_prev_states_(num_prev_states),
      num_next_states_(num_next_states),
      num_variable_prm_(num_variable_prm),
      num_prm_(num_variable_prm + fixed_prm.size()),
      fixed_prm_(std::move(fixed_prm))
  {
    checkDimensions();
  }

  EnzymeTimeVolumeKernel(const FESpace&         space,
                         const GaussQuadrature& quadrature,
                         Index                  num_res,
                         Index                  num_prev_states,
                         Index                  num_next_states,
                         Vector<Real>           fixed_prm)
    : EnzymeTimeVolumeKernel(space,
                             quadrature,
                             num_res,
                             num_prev_states,
                             num_next_states,
                             0,
                             std::move(fixed_prm))
  {
  }

  void checkDimensions() const
  {
    if (num_res_ < 0 || num_prev_states_ < 0
        || num_next_states_ < 0 || num_variable_prm_ < 0 || num_prm_ < 0)
    {
      throw std::runtime_error(
          "EnzymeTimeVolumeKernel received invalid dimensions");
    }
  }

  void res(Index               step,
           Index               ic,
           const Vector<Real>& prev_state,
           const Vector<Real>& next_state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    checkInputSizes(prev_state, next_state, prm);
    resize(out, num_res_);
    const Vector<Real> residual_prm = makeResidualPrm(prm);

    ElementValues values(space_.finiteElement(), quad_);
    values.reinit(space_.mesh().cell(ic));
    Residual(step,
             ic,
             values.numQuadraturePoints(),
             values.numNodes(),
             values.dim(),
             num_res_,
             num_prev_states_,
             num_next_states_,
             num_prm_,
             values.NData(),
             values.dNdxData(),
             values.JxWData(),
             prev_state.data(),
             next_state.data(),
             residual_prm.data(),
             out.data());
  }

  void jacobian(Index                  step,
                Index                  ic,
                problem::VariableBlock wrt,
                const Vector<Real>&    prev_state,
                const Vector<Real>&    next_state,
                const Vector<Real>&    prm,
                DenseMatrix&           out) const override
  {
    checkInputSizes(prev_state, next_state, prm);

    if (wrt == problem::VariableBlock::PrevState)
    {
      PrevStateJac(step, ic, prev_state, next_state, prm, out);
      return;
    }
    if (wrt == problem::VariableBlock::NextState)
    {
      nextStateJac(step, ic, prev_state, next_state, prm, out);
      return;
    }
    paramJac(step, ic, prev_state, next_state, prm, out);
  }

private:
  void PrevStateJac(Index               step,
                        Index               ic,
                        const Vector<Real>& prev_state,
                        const Vector<Real>& next_state,
                        const Vector<Real>& prm,
                        DenseMatrix&        out) const
  {
    out.resize(num_res_, num_prev_states_);
    if (num_prev_states_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      values(space_.finiteElement(), quad_);
    values.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(num_res_);
    Vector<Real> out_adj(num_res_);
    Vector<Real> previous_adj(num_prev_states_);

    for (Index row = 0; row < num_res_; ++row)
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
                              values.numQuadraturePoints(),
                              enzyme_const,
                              values.numNodes(),
                              enzyme_const,
                              values.dim(),
                              enzyme_const,
                              num_res_,
                              enzyme_const,
                              num_prev_states_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              num_prm_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.dNdxData(),
                              enzyme_const,
                              values.JxWData(),
                              enzyme_dup,
                              prev_state.data(),
                              previous_adj.data(),
                              enzyme_const,
                              next_state.data(),
                              enzyme_const,
                              residual_prm.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < num_prev_states_; ++col)
      {
        out(row, col) = previous_adj[col];
      }
    }
#else
    (void) step;
    (void) ic;
    (void) prev_state;
    (void) next_state;
    (void) prm;
    throwUnavailable();
#endif
  }

  void nextStateJac(Index               step,
                    Index               ic,
                    const Vector<Real>& prev_state,
                    const Vector<Real>& next_state,
                    const Vector<Real>& prm,
                    DenseMatrix&        out) const
  {
    out.resize(num_res_, num_next_states_);
    if (num_next_states_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      values(space_.finiteElement(), quad_);
    values.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(num_res_);
    Vector<Real> out_adj(num_res_);
    Vector<Real> next_adj(num_next_states_);

    for (Index row = 0; row < num_res_; ++row)
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
                              values.numQuadraturePoints(),
                              enzyme_const,
                              values.numNodes(),
                              enzyme_const,
                              values.dim(),
                              enzyme_const,
                              num_res_,
                              enzyme_const,
                              num_prev_states_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              num_prm_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.dNdxData(),
                              enzyme_const,
                              values.JxWData(),
                              enzyme_const,
                              prev_state.data(),
                              enzyme_dup,
                              next_state.data(),
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
    (void) prev_state;
    (void) next_state;
    (void) prm;
    throwUnavailable();
#endif
  }

  void paramJac(Index               step,
                Index               ic,
                const Vector<Real>& prev_state,
                const Vector<Real>& next_state,
                const Vector<Real>& prm,
                DenseMatrix&        out) const
  {
    out.resize(num_res_, num_variable_prm_);
    if (num_variable_prm_ == 0)
    {
      return;
    }

#if defined(FEMX_HAS_ENZYME)
    const Vector<Real> residual_prm = makeResidualPrm(prm);
    ElementValues      values(space_.finiteElement(), quad_);
    values.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(num_res_);
    Vector<Real> out_adj(num_res_);
    Vector<Real> param_adj(num_prm_);

    for (Index row = 0; row < num_res_; ++row)
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
                              values.numQuadraturePoints(),
                              enzyme_const,
                              values.numNodes(),
                              enzyme_const,
                              values.dim(),
                              enzyme_const,
                              num_res_,
                              enzyme_const,
                              num_prev_states_,
                              enzyme_const,
                              num_next_states_,
                              enzyme_const,
                              num_prm_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.dNdxData(),
                              enzyme_const,
                              values.JxWData(),
                              enzyme_const,
                              prev_state.data(),
                              enzyme_const,
                              next_state.data(),
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
    (void) prev_state;
    (void) next_state;
    (void) prm;
    throwUnavailable();
#endif
  }

  void checkInputSizes(const Vector<Real>& prev_state,
                       const Vector<Real>& next_state,
                       const Vector<Real>& prm) const
  {
    if (prev_state.size() != num_prev_states_
        || next_state.size() != num_next_states_
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

    Vector<Real> out(num_prm_);
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

  static void resize(Vector<Real>& out, Index size)
  {
    if (out.size() != size)
    {
      out.resize(size);
    }
    else
    {
      out.setZero();
    }
  }

  [[noreturn]] static void throwUnavailable()
  {
    throw std::runtime_error(
        "EnzymeTimeVolumeKernel requires FEMX_ENABLE_ENZYME=ON");
  }

private:
  const FESpace&         space_;
  const GaussQuadrature& quad_;
  Index                  num_res_{0};
  Index                  num_prev_states_{0};
  Index                  num_next_states_{0};
  Index                  num_variable_prm_{0};
  Index                  num_prm_{0};
  Vector<Real>           fixed_prm_;
};

} // namespace assembly
} // namespace femx
