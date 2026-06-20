#pragma once

#include <stdexcept>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/core/Types.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace assembly
{

using LocalVolumeResidualFunction =
    void (*)(Index       cell,
             Index       num_qp,
             Index       num_nodes,
             Index       dim,
             Index       num_res,
             Index       num_states,
             Index       num_prm,
             const Real* N,
             const Real* dNdx,
             const Real* JxW,
             const Real* state,
             const Real* prm,
             Real*       out);

inline bool canUseEnzymeVolumeKernel(const ElementValues& values,
                                     Index                local_size,
                                     Index                max_nodes,
                                     Index                max_local_size)
{
#if defined(FEMX_HAS_ENZYME)
  return values.numDofs() <= max_nodes && values.dim() <= 3
         && local_size <= max_local_size;
#else
  (void) values;
  (void) local_size;
  (void) max_nodes;
  (void) max_local_size;
  return false;
#endif
}

/** @brief Enzyme element kernel with volume element values. */
template <LocalVolumeResidualFunction Residual>
class EnzymeVolumeKernel final : public ElementKernel
{
public:
  EnzymeVolumeKernel(const FESpace&         space,
                     const GaussQuadrature& quadrature,
                     Index                  num_res,
                     Index                  num_states,
                     Index                  num_prm)
    : space_(space),
      quad_(quadrature),
      num_res_(num_res),
      num_states_(num_states),
      num_prm_(num_prm)
  {
    if (num_res_ < 0 || num_states_ < 0 || num_prm_ < 0)
    {
      throw std::runtime_error(
          "EnzymeVolumeKernel received invalid dimensions");
    }
  }

  void res(Index               ic,
           const Vector<Real>& u,
           const Vector<Real>& m,
           Vector<Real>&       out) const override
  {
    checkInputSizes(u, m);
    resize(out, num_res_);

    ElementValues values(space_.finiteElement(), quad_);
    values.reinit(space_.mesh().cell(ic));
    Residual(ic,
             values.numQuadraturePoints(),
             values.numNodes(),
             values.dim(),
             num_res_,
             num_states_,
             num_prm_,
             values.NData(),
             values.dNdxData(),
             values.JxWData(),
             u.data(),
             m.data(),
             out.data());
  }

  void stateJac(Index               ic,
                const Vector<Real>& u,
                const Vector<Real>& m,
                DenseMatrix&        out) const override
  {
    checkInputSizes(u, m);
    out.resize(num_res_, num_states_);

#if defined(FEMX_HAS_ENZYME)
    ElementValues values(space_.finiteElement(), quad_);
    values.reinit(space_.mesh().cell(ic));

    Vector<Real> primal_out(num_res_);
    Vector<Real> out_adj(num_res_);
    Vector<Real> state_adj(num_states_);

    for (Index row = 0; row < num_res_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      state_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
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
                              num_states_,
                              enzyme_const,
                              num_prm_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.dNdxData(),
                              enzyme_const,
                              values.JxWData(),
                              enzyme_dup,
                              u.data(),
                              state_adj.data(),
                              enzyme_const,
                              m.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < num_states_; ++col)
      {
        out(row, col) = state_adj[col];
      }
    }
#else
    (void) ic;
    (void) out;
    throwUnavailable();
#endif
  }

  void paramJac(Index               ic,
                const Vector<Real>& u,
                const Vector<Real>& m,
                DenseMatrix&        out) const override
  {
    checkInputSizes(u, m);
    out.resize(num_res_, num_prm_);

#if defined(FEMX_HAS_ENZYME)
    ElementValues values(space_.finiteElement(), quad_);
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
                              num_states_,
                              enzyme_const,
                              num_prm_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.dNdxData(),
                              enzyme_const,
                              values.JxWData(),
                              enzyme_const,
                              u.data(),
                              enzyme_dup,
                              m.data(),
                              param_adj.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < num_prm_; ++col)
      {
        out(row, col) = param_adj[col];
      }
    }
#else
    (void) ic;
    (void) out;
    throwUnavailable();
#endif
  }

private:
  void checkInputSizes(const Vector<Real>& u, const Vector<Real>& m) const
  {
    if (u.size() != num_states_ || m.size() != num_prm_)
    {
      throw std::runtime_error("EnzymeVolumeKernel input size mismatch");
    }
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
        "EnzymeVolumeKernel requires FEMX_ENABLE_ENZYME=ON");
  }

private:
  const FESpace&         space_;
  const GaussQuadrature& quad_;
  Index                  num_res_{0};
  Index                  num_states_{0};
  Index                  num_prm_{0};
};

} // namespace assembly
} // namespace femx
