#pragma once

#include <stdexcept>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/BoundaryElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/BoundaryElementValues.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

using LocalBoundaryResidualFunction =
    void (*)(Index       facet,
             Index       nq,
             Index       nn,
             Index       dim,
             Index       nres,
             Index       nst,
             Index       nprm,
             const Real* N,
             const Real* point,
             const Real* nrm,
             const Real* JxW,
             const Real* state,
             const Real* prm,
             Real*       out);

/** @brief Enzyme boundary kernel with boundary element values. */
template <LocalBoundaryResidualFunction Residual>
class EnzymeBoundaryKernel final : public BoundaryElementKernel
{
public:
  EnzymeBoundaryKernel(const Mesh&            mesh,
                       const GaussQuadrature& quadrature,
                       Index                  nres,
                       Index                  nst,
                       Index                  nprm)
    : mesh_(mesh),
      quad_(quadrature),
      nres_(nres),
      nst_(nst),
      nprm_(nprm)
  {
    if (nres_ < 0 || nst_ < 0 || nprm_ < 0)
    {
      throw std::runtime_error(
          "EnzymeBoundaryKernel received invalid dimensions");
    }
  }

  void res(Index                      ib,
           const Mesh::BoundaryFacet& facet,
           const Vector<Real>&        u,
           const Vector<Real>&        m,
           Vector<Real>&              out) const override
  {
    checkInputSizes(u, m);
    resizeOrZero(out, nres_);

    BoundaryElementValues vals(quad_);
    vals.reinit(mesh_, facet);
    Residual(ib,
             vals.numQuadraturePoints(),
             vals.numNodes(),
             vals.dim(),
             nres_,
             nst_,
             nprm_,
             vals.NData(),
             vals.pointData(),
             vals.normalData(),
             vals.JxWData(),
             u.data(),
             m.data(),
             out.data());
  }

  void stateJac(Index                      ib,
                const Mesh::BoundaryFacet& facet,
                const Vector<Real>&        u,
                const Vector<Real>&        m,
                DenseMatrix&               out) const override
  {
    checkInputSizes(u, m);
    out.resize(nres_, nst_);

#if defined(FEMX_HAS_ENZYME)
    BoundaryElementValues vals(quad_);
    vals.reinit(mesh_, facet);

    Vector<Real> primal_out(nres_);
    Vector<Real> out_adj(nres_);
    Vector<Real> state_adj(nst_);

    for (Index row = 0; row < nres_; ++row)
    {
      primal_out.setZero();
      out_adj.setZero();
      state_adj.setZero();
      out_adj[row] = 1.0;

      __enzyme_autodiff<void>(reinterpret_cast<void*>(Residual),
                              enzyme_const,
                              ib,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              nres_,
                              enzyme_const,
                              nst_,
                              enzyme_const,
                              nprm_,
                              enzyme_const,
                              vals.NData(),
                              enzyme_const,
                              vals.pointData(),
                              enzyme_const,
                              vals.normalData(),
                              enzyme_const,
                              vals.JxWData(),
                              enzyme_dup,
                              u.data(),
                              state_adj.data(),
                              enzyme_const,
                              m.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < nst_; ++col)
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

  void paramJac(Index                      ib,
                const Mesh::BoundaryFacet& facet,
                const Vector<Real>&        u,
                const Vector<Real>&        m,
                DenseMatrix&               out) const override
  {
    checkInputSizes(u, m);
    out.resize(nres_, nprm_);

#if defined(FEMX_HAS_ENZYME)
    BoundaryElementValues vals(quad_);
    vals.reinit(mesh_, facet);

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
                              ib,
                              enzyme_const,
                              vals.numQuadraturePoints(),
                              enzyme_const,
                              vals.numNodes(),
                              enzyme_const,
                              vals.dim(),
                              enzyme_const,
                              nres_,
                              enzyme_const,
                              nst_,
                              enzyme_const,
                              nprm_,
                              enzyme_const,
                              vals.NData(),
                              enzyme_const,
                              vals.pointData(),
                              enzyme_const,
                              vals.normalData(),
                              enzyme_const,
                              vals.JxWData(),
                              enzyme_const,
                              u.data(),
                              enzyme_dup,
                              m.data(),
                              param_adj.data(),
                              enzyme_dup,
                              primal_out.data(),
                              out_adj.data());

      for (Index col = 0; col < nprm_; ++col)
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
  void checkInputSizes(const Vector<Real>& u, const Vector<Real>& m) const
  {
    if (u.size() != nst_ || m.size() != nprm_)
    {
      throw std::runtime_error("EnzymeBoundaryKernel input size mismatch");
    }
  }

  [[noreturn]] static void throwUnavailable()
  {
    throw std::runtime_error(
        "EnzymeBoundaryKernel requires FEMX_ENABLE_ENZYME=ON");
  }

private:
  const Mesh&            mesh_;
  const GaussQuadrature& quad_;
  Index                  nres_{0};
  Index                  nst_{0};
  Index                  nprm_{0};
};

} // namespace assembly
} // namespace femx
