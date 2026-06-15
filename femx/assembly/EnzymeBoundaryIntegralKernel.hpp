#pragma once

#include <stdexcept>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/BoundaryElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/BoundaryElementValues.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{
namespace assembly
{

using LocalBoundaryIntegralResidualFunction =
    void (*)(Index       facet,
             Index       num_qp,
             Index       num_nodes,
             Index       dim,
             Index       num_res,
             Index       num_states,
             Index       num_params,
             const Real* N,
             const Real* point,
             const Real* normal,
             const Real* JxW,
             const Real* state,
             const Real* params,
             Real*       out);

/** @brief Enzyme boundary kernel with boundary element values. */
template <LocalBoundaryIntegralResidualFunction Residual>
class EnzymeBoundaryIntegralKernel final : public BoundaryElementKernel
{
public:
  EnzymeBoundaryIntegralKernel(const Mesh&            mesh,
                               const GaussQuadrature& quadrature,
                               Index                  num_res,
                               Index                  num_states,
                               Index                  num_params)
    : mesh_(mesh),
      quad_(quadrature),
      num_res_(num_res),
      num_states_(num_states),
      num_params_(num_params)
  {
    if (num_res_ < 0 || num_states_ < 0 || num_params_ < 0)
    {
      throw std::runtime_error(
          "EnzymeBoundaryIntegralKernel received invalid dimensions");
    }
  }

  void res(Index                       ib,
           const Mesh::BoundaryFacet& facet,
           const Vector&              u,
           const Vector&              m,
           Vector&                    out) const override
  {
    checkInputSizes(u, m);
    resize(out, num_res_);

    BoundaryElementValues values(quad_);
    values.reinit(mesh_, facet);
    Residual(ib,
             values.numQuadraturePoints(),
             values.numNodes(),
             values.dim(),
             num_res_,
             num_states_,
             num_params_,
             values.NData(),
             values.pointData(),
             values.normalData(),
             values.JxWData(),
             u.data(),
             m.data(),
             out.data());
  }

  void stateJac(Index                       ib,
                const Mesh::BoundaryFacet& facet,
                const Vector&              u,
                const Vector&              m,
                DenseMatrix&               out) const override
  {
    checkInputSizes(u, m);
    out.resize(num_res_, num_states_);

#if defined(FEMX_HAS_ENZYME)
    BoundaryElementValues values(quad_);
    values.reinit(mesh_, facet);

    Vector primal_out(num_res_);
    Vector out_adj(num_res_);
    Vector state_adj(num_states_);

    for (Index row = 0; row < num_res_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      state_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              ib,
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
                              num_params_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.pointData(),
                              enzyme_const,
                              values.normalData(),
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
    (void) ib;
    (void) facet;
    (void) out;
    throwUnavailable();
#endif
  }

  void paramJac(Index                       ib,
                const Mesh::BoundaryFacet& facet,
                const Vector&              u,
                const Vector&              m,
                DenseMatrix&               out) const override
  {
    checkInputSizes(u, m);
    out.resize(num_res_, num_params_);

#if defined(FEMX_HAS_ENZYME)
    BoundaryElementValues values(quad_);
    values.reinit(mesh_, facet);

    Vector primal_out(num_res_);
    Vector out_adj(num_res_);
    Vector param_adj(num_params_);

    for (Index row = 0; row < num_res_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      param_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              ib,
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
                              num_params_,
                              enzyme_const,
                              values.NData(),
                              enzyme_const,
                              values.pointData(),
                              enzyme_const,
                              values.normalData(),
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

      for (Index col = 0; col < num_params_; ++col)
      {
        out(row, col) = param_adj[col];
      }
    }
#else
    (void) ib;
    (void) facet;
    (void) out;
    throwUnavailable();
#endif
  }

private:
  void checkInputSizes(const Vector& u, const Vector& m) const
  {
    if (u.size() != num_states_ || m.size() != num_params_)
    {
      throw std::runtime_error(
          "EnzymeBoundaryIntegralKernel input size mismatch");
    }
  }

  static void resize(Vector& out, Index size)
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
        "EnzymeBoundaryIntegralKernel requires FEMX_ENABLE_ENZYME=ON");
  }

private:
  const Mesh&            mesh_;
  const GaussQuadrature& quad_;
  Index                  num_res_{0};
  Index                  num_states_{0};
  Index                  num_params_{0};
};

} // namespace assembly
} // namespace femx
