#include "NavierKernel.hpp"

#include <stdexcept>
#include <utility>

#include "Components.hpp"
#include <femx/fem/ElementValues.hpp>

using namespace std;

namespace femx::navier
{
namespace
{

bool supportedLocalSize(Index nq,
                        Index nn,
                        Index dim,
                        Index nloc)
{
  return dim > 0 && dim <= kMaxSpatialDim
         && nn > 0 && nn <= kMaxLocalNodes
         && nq > 0 && nq <= kMaxLocalQuadraturePoints
         && nloc > 0 && nloc <= kMaxLocalDofs
         && nloc == numLocalDofs(nn, dim);
}

Index historyDepth(Index num_prev_states, Index nloc)
{
  if (nloc <= 0 || num_prev_states < nloc
      || num_prev_states % nloc != 0)
  {
    return 0;
  }
  return num_prev_states / nloc;
}

bool assembleLocalSystem(Index       step,
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
                         const Real* prm,
                         LocalMatrix Ke,
                         LocalVector Fe)
{
  if (!supportedLocalSize(nq, nn, dim, nres)
      || num_next_states != nres
      || historyDepth(num_prev_states, nres) <= 0 || nprm < 3)
  {
    return false;
  }

  const Index       nd = numLocalDofs(nn, dim);
  const KernelFluid fluid{prm[0], prm[1]};
  const Real        dt = prm[2];

  LocalElementValues ev{nq, nn, dim, N, dNdx, JxW};
  QPState            qps[64]{};
  Real               adv[64];

  const Real* cur = prev;
  const Real* old = prev + nd;
  for (Index i = 0; i < nd; ++i)
  {
    adv[i] = cur[i];
  }
  if (step > 0 && historyDepth(num_prev_states, nd) >= 2)
  {
    for (Index i = 0; i < nd; ++i)
    {
      adv[i] = 1.5 * cur[i] - 0.5 * old[i];
    }
  }

  updateQpStates(ev, fluid, dt, cur, adv, qps);
  zeroLocalSystem(nd, Ke, Fe);

  assembleMassLHS(ev, fluid, dt, Ke);
  assembleAdvectionLHS(ev, qps, fluid, Ke);
  assembleDiffusionLHS(ev, fluid, Ke);
  assemblePreVelCouplingLHS(ev, Ke);
  assembleStabilizationLHS(ev, qps, fluid, dt, Ke);

  assembleMassRHS(ev, qps, fluid, dt, Fe);
  assembleAdvectionRHS(ev, qps, fluid, Fe);
  assembleDiffusionRHS(ev, qps, fluid, Fe);
  assembleStabilizationRHS(ev, qps, fluid, dt, Fe);
  return true;
}

} // namespace

void NavierResidual(Index       step,
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
                    Real*       out)
{
  (void) cell;

  for (Index i = 0; i < nres; ++i)
  {
    out[i] = 0.0;
  }

  (void) nprm;
  const Index nd = numLocalDofs(nn, dim);
  Real        s1[64 * 64];
  Real        s2[64];
  LocalMatrix Ke{nd, s1};
  LocalVector Fe{nd, s2};

  if (!assembleLocalSystem(step,
                           nq,
                           nn,
                           dim,
                           nres,
                           num_prev_states,
                           num_next_states,
                           nprm,
                           N,
                           dNdx,
                           JxW,
                           prev,
                           prm,
                           Ke,
                           Fe))
  {
    return;
  }

  finishLocalResidual(nd, nxt, Ke, Fe, out);
}

NavierKernel::NavierKernel(const FESpace&         space,
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
    fixed_prm_(fixed_prm),
    fallback_(space,
              quadrature,
              nres,
              num_prev_states,
              num_next_states,
              num_variable_prm,
              std::move(fixed_prm))
{
  checkDimensions();
}

NavierKernel::NavierKernel(const FESpace&         space,
                           const GaussQuadrature& quadrature,
                           Index                  nres,
                           Index                  num_prev_states,
                           Index                  num_next_states,
                           Vector<Real>           fixed_prm)
  : NavierKernel(space,
                 quadrature,
                 nres,
                 num_prev_states,
                 num_next_states,
                 0,
                 std::move(fixed_prm))
{
}

void NavierKernel::res(Index                    step,
                       Index                    ic,
                       problem::TimeHistoryView hist,
                       const Vector<Real>&      nxt,
                       const Vector<Real>&      prm,
                       Vector<Real>&            out) const
{
  checkInputSizes(hist, nxt, prm);
  resizeOrZero(out, nres_);

  ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().cell(ic));

  const Index nd = num_next_states_;
  Real        s1[64 * 64];
  Real        s2[64];
  LocalMatrix Ke{nd, s1};
  LocalVector Fe{nd, s2};

  Vector<Real> residual_prm;
  const Real*  residual_prm_data = fixed_prm_.data();
  if (num_variable_prm_ > 0)
  {
    residual_prm      = makeResidualPrm(prm);
    residual_prm_data = residual_prm.data();
  }

  if (!assembleLocalSystem(step,
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
                           residual_prm_data,
                           Ke,
                           Fe))
  {
    return;
  }

  finishLocalResidual(nd, nxt.data(), Ke, Fe, out.data());
}

void NavierKernel::jacobian(Index                    step,
                            Index                    ic,
                            problem::VariableBlock   wrt,
                            problem::TimeHistoryView hist,
                            const Vector<Real>&      nxt,
                            const Vector<Real>&      prm,
                            DenseMatrix&             out) const
{
  checkInputSizes(hist, nxt, prm);
  if (!wrt.isNextState())
  {
    fallback_.jacobian(step, ic, wrt, hist, nxt, prm, out);
    return;
  }

  out.resize(nres_, num_next_states_);

  ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().cell(ic));

  const Index nd = num_next_states_;
  Real        s1[64 * 64];
  Real        s2[64];
  LocalMatrix Ke{nd, s1};
  LocalVector Fe{nd, s2};

  Vector<Real> residual_prm;
  const Real*  residual_prm_data = fixed_prm_.data();
  if (num_variable_prm_ > 0)
  {
    residual_prm      = makeResidualPrm(prm);
    residual_prm_data = residual_prm.data();
  }

  if (!assembleLocalSystem(step,
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
                           residual_prm_data,
                           Ke,
                           Fe))
  {
    out.setZero();
    return;
  }

  for (Index i = 0; i < nres_; ++i)
  {
    for (Index j = 0; j < num_next_states_; ++j)
    {
      out(i, j) = Ke(i, j);
    }
  }
}

void NavierKernel::checkDimensions()
{
  if (nres_ < 0 || num_prev_states_ < 0
      || num_next_states_ < 0 || num_variable_prm_ < 0 || nprm_ < 0)
  {
    throw runtime_error("NavierKernel received invalid dimensions");
  }
  if (num_next_states_ <= 0 || num_prev_states_ % num_next_states_ != 0)
  {
    throw runtime_error(
        "NavierKernel history size must be a multiple of next-state size");
  }
  num_history_states_     = num_prev_states_ / num_next_states_;
  num_history_state_dofs_ = num_next_states_;
  if (nres_ != num_next_states_ || nprm_ < 3)
  {
    throw runtime_error("NavierKernel received unsupported dimensions");
  }
}

void NavierKernel::checkInputSizes(problem::TimeHistoryView hist,
                                   const Vector<Real>&      nxt,
                                   const Vector<Real>&      prm) const
{
  if (hist.count() != num_history_states_
      || hist.stateSize() != num_history_state_dofs_
      || nxt.size() != num_next_states_
      || prm.size() != num_variable_prm_)
  {
    throw runtime_error("NavierKernel input size mismatch");
  }
}

Vector<Real> NavierKernel::makeResidualPrm(
    const Vector<Real>& variable_prm) const
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

Vector<Real> physicalParams(Real rho, Real mu, Real dt)
{
  if (rho <= 0.0 || mu < 0.0 || dt <= 0.0)
  {
    throw runtime_error("NavierKernel received invalid parameters");
  }

  Vector<Real> prm(3);
  prm[0] = rho;
  prm[1] = mu;
  prm[2] = dt;
  return prm;
}

NavierKernel makeNavierKernel(const FESpace&         velocity_space,
                              const GaussQuadrature& quadrature,
                              Index                  nloc,
                              Real                   rho,
                              Real                   mu,
                              Real                   dt)
{
  if (!supportedLocalSize(quadrature.size(),
                          velocity_space.numShapesPerElem(),
                          velocity_space.numComponents(),
                          nloc))
  {
    throw runtime_error("NavierKernel received unsupported element size");
  }

  return NavierKernel(velocity_space,
                      quadrature,
                      nloc,
                      2 * nloc,
                      nloc,
                      physicalParams(rho, mu, dt));
}

} // namespace femx::navier
