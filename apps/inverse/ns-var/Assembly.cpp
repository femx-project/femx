#include "Assembly.hpp"

#include <stdexcept>

#include "Components.hpp"
#include <femx/assembly/EnzymeVolumeKernel.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/Quadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>

using namespace femx::assembly;

namespace femx
{

namespace
{

constexpr Index kMaxElemNodes = 8;
constexpr Index kMaxElemDofs  = 32;
constexpr Index kMaxElemQps   = 64;

void zeroElemSystem(Index num_dofs, Real* Ke, Real* Fe)
{
  for (Index i = 0; i < num_dofs; ++i)
  {
    Fe[i] = 0.0;
    for (Index j = 0; j < num_dofs; ++j)
    {
      Ke[i * num_dofs + j] = 0.0;
    }
  }
}

void finish(Index num_dofs, const Real* Ke, const Real* Fe, const Real* x_next, Real* out)
{
  for (Index i = 0; i < num_dofs; ++i)
  {
    Real value = -Fe[i];
    for (Index j = 0; j < num_dofs; ++j)
    {
      value += Ke[i * num_dofs + j] * x_next[j];
    }
    out[i] = value;
  }
}

void NavierVolumeResidual(Index       cell,
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
                          Real*       out)
{
  (void) cell;
  (void) num_res;
  (void) num_prm;

  const Index num_shape = num_nodes;
  const Index num_dofs  = dim * num_shape + num_shape;
  const Real* x_next    = prm;
  const Real  rho       = prm[num_states];
  const Real  mu        = prm[num_states + 1];
  const Real  dt        = prm[num_states + 2];

  Qp   qps[kMaxElemQps]{};
  Real Ke[kMaxElemDofs * kMaxElemDofs];
  Real Fe[kMaxElemDofs];

  updateElemState(qps,
                  num_qp,
                  num_shape,
                  dim,
                  N,
                  dNdx,
                  state,
                  rho,
                  mu,
                  dt);

  zeroElemSystem(num_dofs, Ke, Fe);

  for (Index iq = 0; iq < num_qp; ++iq)
  {
    const Real* Nq     = N + iq * num_shape;
    const Real* dNdx_q = dNdx + iq * num_shape * dim;
    const Qp&   qp     = qps[iq];

    assembleMassLHS(num_shape, dim, rho, dt, Nq, JxW[iq], Ke);
    assembleAdvectionLHS(num_shape, dim, rho, Nq, dNdx_q, JxW[iq], qp, Ke);
    assembleDiffusionLHS(num_shape, dim, mu, dNdx_q, JxW[iq], Ke);
    assemblePressureLHS(num_shape, dim, Nq, dNdx_q, JxW[iq], Ke);
    assembleStabilizationLHS(num_shape, dim, rho, dt, Nq, dNdx_q, JxW[iq], qp, Ke);

    assembleMassRHS(num_shape, dim, rho, dt, Nq, JxW[iq], qp, Fe);
    assembleAdvectionRHS(num_shape, dim, rho, Nq, JxW[iq], qp, Fe);
    assembleDiffusionRHS(num_shape, dim, mu, dNdx_q, JxW[iq], qp, Fe);
    assembleStabilizationRHS(num_shape, dim, rho, dt, dNdx_q, JxW[iq], qp, Fe);
  }

  finish(num_dofs, Ke, Fe, x_next, out);
}

using Kernel = EnzymeVolumeKernel<NavierVolumeResidual>;

void checkSize(const ElementValues& values, Index num_dofs)
{
  if (values.numDofs() > kMaxElemNodes
      || values.numQuadraturePoints() > kMaxElemQps || values.dim() > 3
      || num_dofs > kMaxElemDofs)
  {
    throw std::runtime_error("Navier volume kernel element is too large");
  }
}

void checkEnzyme(const ElementValues& values, Index num_dofs)
{
  if (values.numQuadraturePoints() > kMaxElemQps
      || !canUseEnzymeVolumeKernel(
          values, num_dofs, kMaxElemNodes, kMaxElemDofs))
  {
    throw std::runtime_error(
        "Navier state Jacobian requires FEMX_ENABLE_ENZYME=ON and a supported "
        "element");
  }
}

void gather(const MixedFESpace& space,
            Index               cell,
            const Vector<Real>& global,
            Vector<Real>&       local)
{
  const Index num_dofs = space.numDofsPerElem();
  if (local.size() != num_dofs)
  {
    local.resize(num_dofs);
  }

  Vector<Index> dofs;
  space.elemDofs(cell, dofs);
  for (Index i = 0; i < num_dofs; ++i)
  {
    local[i] = global[dofs[i]];
  }
}

void pack(const MixedFESpace&               space,
          Index                             cell,
          const Vector<Real>&               x_next,
          const TimeNavierStokesParameters& prm,
          Vector<Real>&                     out)
{
  const Index num_dofs = space.numDofsPerElem();
  if (out.size() != num_dofs + 3)
  {
    out.resize(num_dofs + 3);
  }

  Vector<Index> dofs;
  space.elemDofs(cell, dofs);
  for (Index i = 0; i < num_dofs; ++i)
  {
    out[i] = x_next[dofs[i]];
  }
  out[num_dofs]     = prm.fluid.rho;
  out[num_dofs + 1] = prm.fluid.mu;
  out[num_dofs + 2] = prm.dt;
}

void inputs(const MixedFESpace&               space,
            Index                             cell,
            const Vector<Real>&               x_next,
            const Vector<Real>&               x,
            const TimeNavierStokesParameters& prm,
            Vector<Real>&                     state,
            Vector<Real>&                     kernel_prm)
{
  gather(space, cell, x, state);
  pack(space, cell, x_next, prm, kernel_prm);
}

} // namespace

void assembleElemResidual(const MixedFESpace&               space,
                          Index                             cell,
                          ElementValues&                    values,
                          const Vector<Real>&               x_next,
                          const Vector<Real>&               x,
                          const TimeNavierStokesParameters& prm,
                          Vector<Real>&                     out)
{
  const Index num_dofs = space.numDofsPerElem();
  values.reinit(space.mesh().cell(cell));
  checkSize(values, num_dofs);

  Vector<Real> state;
  Vector<Real> kernel_prm;
  inputs(space, cell, x_next, x, prm, state, kernel_prm);

  if (out.size() != num_dofs)
  {
    out.resize(num_dofs);
  }

  NavierVolumeResidual(cell,
                       values.numQuadraturePoints(),
                       values.numNodes(),
                       values.dim(),
                       num_dofs,
                       num_dofs,
                       num_dofs + 3,
                       values.NData(),
                       values.dNdxData(),
                       values.JxWData(),
                       state.data(),
                       kernel_prm.data(),
                       out.data());
}

void assembleNextElemMatrix(const MixedFESpace&               space,
                            Index                             cell,
                            ElementValues&                    values,
                            const GaussQuadrature&            quad,
                            const Vector<Real>&               x_next,
                            const Vector<Real>&               x,
                            const TimeNavierStokesParameters& prm,
                            DenseMatrix&                      out)
{
  const Index num_dofs = space.numDofsPerElem();
  values.reinit(space.mesh().cell(cell));
  checkEnzyme(values, num_dofs);

  Vector<Real> state;
  Vector<Real> kernel_prm;
  inputs(space, cell, x_next, x, prm, state, kernel_prm);

  DenseMatrix full;
  Kernel      kernel(space.field(0).space(), quad, num_dofs, num_dofs, num_dofs + 3);
  kernel.paramJac(cell, state, kernel_prm, full);

  out.resize(num_dofs, num_dofs);
  for (Index i = 0; i < num_dofs; ++i)
  {
    for (Index j = 0; j < num_dofs; ++j)
    {
      out(i, j) = full(i, j);
    }
  }
}

void assemblePrevElemMatrix(const MixedFESpace&               space,
                            Index                             cell,
                            ElementValues&                    values,
                            const GaussQuadrature&            quad,
                            const Vector<Real>&               x_next,
                            const Vector<Real>&               x,
                            const TimeNavierStokesParameters& prm,
                            DenseMatrix&                      out)
{
  const Index num_dofs = space.numDofsPerElem();
  values.reinit(space.mesh().cell(cell));
  checkEnzyme(values, num_dofs);

  Vector<Real> state;
  Vector<Real> kernel_prm;
  inputs(space, cell, x_next, x, prm, state, kernel_prm);

  Kernel kernel(space.field(0).space(), quad, num_dofs, num_dofs, num_dofs + 3);
  kernel.stateJac(cell, state, kernel_prm, out);
}

} // namespace femx
