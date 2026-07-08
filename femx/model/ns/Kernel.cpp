#include "Kernel.hpp"

#include <stdexcept>
#include <utility>

#include "Components.hpp"
#include <femx/fem/ElementValues.hpp>

using namespace std;

namespace femx::model::ns
{
namespace
{

bool supportedLocalSize(Index num_qpts,
                        Index num_nodes,
                        Index dim,
                        Index num_local_dofs)
{
  return dim > 0 && dim <= kMaxDim
         && num_nodes > 0 && num_nodes <= kMaxNn
         && num_qpts > 0 && num_qpts <= kMaxNq
         && num_local_dofs > 0 && num_local_dofs <= kMaxNd
         && num_local_dofs == numLocalDofs(num_nodes, dim);
}

Index historyDepth(Index num_history_dofs, Index num_local_dofs)
{
  if (num_local_dofs <= 0 || num_history_dofs < num_local_dofs
      || num_history_dofs % num_local_dofs != 0)
  {
    return 0;
  }
  return num_history_dofs / num_local_dofs;
}

bool assembleLocalSystem(Index       step,
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
                         const Real* prm,
                         LocalMatrix Ke,
                         LocalVector Fe)
{
  if (!supportedLocalSize(num_qpts, num_nodes, dim, num_residuals)
      || num_next_states != num_residuals
      || historyDepth(num_history_dofs, num_residuals) <= 0 || num_params < 3)
  {
    return false;
  }

  const Index       num_dofs = numLocalDofs(num_nodes, dim);
  const KernelFluid fluid{prm[0], prm[1]};
  const Real        dt = prm[2];

  LocalElementValues ev{num_qpts, num_nodes, dim, N, dNdx, JxW};
  QPState            qps[kMaxNq]{};
  Real               adv[kMaxNd];

  const Real* cur = hist;
  const Real* old = hist + num_dofs;
  for (Index i = 0; i < num_dofs; ++i)
  {
    adv[i] = cur[i];
  }
  if (step > 0 && historyDepth(num_history_dofs, num_dofs) >= 2)
  {
    for (Index i = 0; i < num_dofs; ++i)
    {
      adv[i] = 1.5 * cur[i] - 0.5 * old[i];
    }
  }

  updateQpStates(ev, fluid, dt, cur, adv, qps);
  zeroLocalSystem(num_dofs, Ke, Fe);

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
                    Real*       out)
{
  (void) elem;

  for (Index i = 0; i < num_residuals; ++i)
  {
    out[i] = 0.0;
  }

  (void) num_params;
  const Index num_dofs = numLocalDofs(num_nodes, dim);
  Real        s1[kMaxNd * kMaxNd];
  Real        s2[kMaxNd];
  LocalMatrix Ke{num_dofs, s1};
  LocalVector Fe{num_dofs, s2};

  if (!assembleLocalSystem(step,
                           num_qpts,
                           num_nodes,
                           dim,
                           num_residuals,
                           num_history_dofs,
                           num_next_states,
                           num_params,
                           N,
                           dNdx,
                           JxW,
                           hist,
                           prm,
                           Ke,
                           Fe))
  {
    return;
  }

  finishLocalResidual(num_dofs, nxt, Ke, Fe, out);
}

NavierKernel::NavierKernel(const FESpace&         space,
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
    fixed_prm_(fixed_prm),
    enzyme_kernel_(space,
                   quadrature,
                   num_residuals,
                   num_history_dofs,
                   num_next_states,
                   num_variable_params,
                   std::move(fixed_prm))
{
  checkDimensions();
}

NavierKernel::NavierKernel(const FESpace&         space,
                           const GaussQuadrature& quadrature,
                           Index                  num_residuals,
                           Index                  num_history_dofs,
                           Index                  num_next_states,
                           Vector<Real>           fixed_prm)
  : NavierKernel(space,
                 quadrature,
                 num_residuals,
                 num_history_dofs,
                 num_next_states,
                 0,
                 std::move(fixed_prm))
{
}

void NavierKernel::res(Index                    step,
                       Index                    ie,
                       state::TimeHistoryView hist,
                       const Vector<Real>&      nxt,
                       const Vector<Real>&      prm,
                       Vector<Real>&            out) const
{
  checkInputSizes(hist, nxt, prm);
  resizeOrZero(out, num_residuals_);

  ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().elem(ie));

  const Index num_dofs = num_next_states_;
  Real        s1[kMaxNd * kMaxNd];
  Real        s2[kMaxNd];
  LocalMatrix Ke{num_dofs, s1};
  LocalVector Fe{num_dofs, s2};

  Vector<Real> residual_prm;
  const Real*  residual_prm_data = fixed_prm_.data();
  if (num_variable_params_ > 0)
  {
    residual_prm      = makeResidualPrm(prm);
    residual_prm_data = residual_prm.data();
  }

  if (!assembleLocalSystem(step,
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
                           residual_prm_data,
                           Ke,
                           Fe))
  {
    return;
  }

  finishLocalResidual(num_dofs, nxt.data(), Ke, Fe, out.data());
}

void NavierKernel::jacobian(Index                    step,
                            Index                    ie,
                            state::VariableBlock   wrt,
                            state::TimeHistoryView hist,
                            const Vector<Real>&      nxt,
                            const Vector<Real>&      prm,
                            DenseMatrix&             out) const
{
  checkInputSizes(hist, nxt, prm);
  if (!wrt.isNextState())
  {
    enzyme_kernel_.jacobian(step, ie, wrt, hist, nxt, prm, out);
    return;
  }

  out.resize(num_residuals_, num_next_states_);

  ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().elem(ie));

  const Index num_dofs = num_next_states_;
  Real        s1[kMaxNd * kMaxNd];
  Real        s2[kMaxNd];
  LocalMatrix Ke{num_dofs, s1};
  LocalVector Fe{num_dofs, s2};

  Vector<Real> residual_prm;
  const Real*  residual_prm_data = fixed_prm_.data();
  if (num_variable_params_ > 0)
  {
    residual_prm      = makeResidualPrm(prm);
    residual_prm_data = residual_prm.data();
  }

  if (!assembleLocalSystem(step,
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
                           residual_prm_data,
                           Ke,
                           Fe))
  {
    out.setZero();
    return;
  }

  for (Index i = 0; i < num_residuals_; ++i)
  {
    for (Index j = 0; j < num_next_states_; ++j)
    {
      out(i, j) = Ke(i, j);
    }
  }
}

void NavierKernel::checkDimensions()
{
  if (num_residuals_ < 0 || num_hist_dofs_ < 0
      || num_next_states_ < 0 || num_variable_params_ < 0 || num_params_ < 0)
  {
    throw runtime_error("NavierKernel received invalid dimensions");
  }
  if (num_next_states_ <= 0 || num_hist_dofs_ % num_next_states_ != 0)
  {
    throw runtime_error(
        "NavierKernel history size must be a multiple of next-state size");
  }
  num_hist_states_     = num_hist_dofs_ / num_next_states_;
  num_hist_state_dofs_ = num_next_states_;
  if (num_residuals_ != num_next_states_ || num_params_ < 3)
  {
    throw runtime_error("NavierKernel received unsupported dimensions");
  }
}

void NavierKernel::checkInputSizes(state::TimeHistoryView hist,
                                   const Vector<Real>&      nxt,
                                   const Vector<Real>&      prm) const
{
  if (hist.count() != num_hist_states_
      || hist.stateSize() != num_hist_state_dofs_
      || nxt.size() != num_next_states_
      || prm.size() != num_variable_params_)
  {
    throw runtime_error("NavierKernel input size mismatch");
  }
}

Vector<Real> NavierKernel::makeResidualPrm(const Vector<Real>& variable_prm) const
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

NavierKernel makeNavierKernel(const FESpace&         vel_space,
                              const GaussQuadrature& quad,
                              Index                  num_local_dofs,
                              Real                   rho,
                              Real                   mu,
                              Real                   dt)
{
  if (!supportedLocalSize(quad.size(),
                          vel_space.numShapesPerElem(),
                          vel_space.numComponents(),
                          num_local_dofs))
  {
    throw runtime_error("NavierKernel received unsupported element size");
  }

  return NavierKernel(vel_space,
                      quad,
                      num_local_dofs,
                      2 * num_local_dofs,
                      num_local_dofs,
                      physicalParams(rho, mu, dt));
}

} // namespace femx::model::ns
