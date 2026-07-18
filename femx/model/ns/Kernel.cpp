#include "Kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "Components.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/fem/ElementValues.hpp>

namespace femx::model::ns
{
namespace
{

bool supportedSize(Index num_qpts,
                   Index num_nodes,
                   Index dim,
                   Index ndof)
{
  return dim > 0 && dim <= kMaxDim
         && num_nodes > 0 && num_nodes <= kMaxNn
         && num_qpts > 0 && num_qpts <= kMaxNq
         && ndof > 0 && ndof <= kMaxNd
         && ndof == numLocalDofs(num_nodes, dim);
}

Index historyDepth(Index num_hist_dofs, Index ndof)
{
  if (ndof <= 0 || num_hist_dofs < ndof
      || num_hist_dofs % ndof != 0)
  {
    return 0;
  }
  return num_hist_dofs / ndof;
}

Real fdStep(Real val)
{
  return std::cbrt(std::numeric_limits<Real>::epsilon())
         * std::max<Real>(1.0, std::abs(val));
}

bool assembleLocal(Index       step,
                   Index       num_qpts,
                   Index       num_nodes,
                   Index       dim,
                   Index       num_res,
                   Index       num_hist_dofs,
                   Index       num_next,
                   Index       num_prm,
                   const Real* N,
                   const Real* dNdx,
                   const Real* JxW,
                   const Real* hist,
                   const Real* prm,
                   LocalMatrix Ke,
                   LocalVector Fe)
{
  if (!supportedSize(num_qpts, num_nodes, dim, num_res)
      || num_next != num_res
      || historyDepth(num_hist_dofs, num_res) <= 0 || num_prm < 3)
  {
    return false;
  }

  const Index       ndof = numLocalDofs(num_nodes, dim);
  const KernelFluid fluid{prm[0], prm[1]};
  const Real        dt = prm[2];

  LocalElementValues ev{num_qpts, num_nodes, dim, N, dNdx, JxW};
  QPState            qps[kMaxNq]{};
  Real               adv[kMaxNd];

  const Real* cur = hist;
  const Real* old = hist + ndof;
  for (Index i = 0; i < ndof; ++i)
  {
    adv[i] = cur[i];
  }
  if (step > 0 && historyDepth(num_hist_dofs, ndof) >= 2)
  {
    for (Index i = 0; i < ndof; ++i)
    {
      adv[i] = 1.5 * cur[i] - 0.5 * old[i];
    }
  }

  updateQpStates(ev, fluid, dt, cur, adv, qps);
  zeroLocalSystem(ndof, Ke, Fe);

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

void navierRes(Index       step,
               Index       ie,
               Index       num_qpts,
               Index       num_nodes,
               Index       dim,
               Index       num_res,
               Index       num_hist_dofs,
               Index       num_next,
               Index       num_prm,
               const Real* N,
               const Real* dNdx,
               const Real* JxW,
               const Real* hist,
               const Real* nxt,
               const Real* prm,
               Real*       out)
{
  (void) ie;

  for (Index i = 0; i < num_res; ++i)
  {
    out[i] = 0.0;
  }

  (void) num_prm;
  const Index ndof = numLocalDofs(num_nodes, dim);
  Real        s1[kMaxNd * kMaxNd];
  Real        s2[kMaxNd];
  LocalMatrix Ke{ndof, s1};
  LocalVector Fe{ndof, s2};

  if (!assembleLocal(step,
                     num_qpts,
                     num_nodes,
                     dim,
                     num_res,
                     num_hist_dofs,
                     num_next,
                     num_prm,
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

  finishLocalResidual(ndof, nxt, Ke, Fe, out);
}

} // namespace

NavierKernel::NavierKernel(const fem::FESpace&         space,
                           const fem::GaussQuadrature& quad,
                           Index                       num_res,
                           Index                       num_hist_dofs,
                           Index                       num_next,
                           Index                       num_var_prm,
                           HostVector                  fixed_prm)
  : space_(space),
    quad_(quad),
    num_res_(num_res),
    num_hist_dofs_(num_hist_dofs),
    num_next_(num_next),
    num_var_prm_(num_var_prm),
    num_prm_(num_var_prm + fixed_prm.size()),
    fixed_prm_(std::move(fixed_prm))
{
  checkDims();
}

NavierKernel::NavierKernel(const fem::FESpace&         space,
                           const fem::GaussQuadrature& quad,
                           Index                       num_res,
                           Index                       num_hist_dofs,
                           Index                       num_next,
                           HostVector                  fixed_prm)
  : NavierKernel(space,
                 quad,
                 num_res,
                 num_hist_dofs,
                 num_next,
                 0,
                 std::move(fixed_prm))
{
}

void NavierKernel::res(Index                  step,
                       Index                  ie,
                       state::TimeHistoryView hist,
                       const HostVector&      nxt,
                       const HostVector&      prm,
                       HostVector&            out) const
{
  checkSizes(hist, nxt, prm);
  resizeOrZero(out, num_res_);

  fem::ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().elem(ie));

  HostVector  res_prm;
  const Real* prm_data = fixed_prm_.data();
  if (num_var_prm_ > 0)
  {
    res_prm  = makeResPrm(prm);
    prm_data = res_prm.data();
  }

  navierRes(step,
            ie,
            vals.numQuadraturePoints(),
            vals.numNodes(),
            vals.dim(),
            num_res_,
            num_hist_dofs_,
            num_next_,
            num_prm_,
            vals.NData(),
            vals.dNdxData(),
            vals.JxWData(),
            hist.data(),
            nxt.data(),
            prm_data,
            out.data());
}

void NavierKernel::jac(Index                  step,
                       Index                  ie,
                       state::VariableBlock   wrt,
                       state::TimeHistoryView hist,
                       const HostVector&      nxt,
                       const HostVector&      prm,
                       DenseMatrix&           out) const
{
  checkSizes(hist, nxt, prm);
  if (!wrt.isNextState())
  {
#if defined(FEMX_HAS_ENZYME)
    adJac(step, ie, wrt, hist, nxt, prm, out);
#else
    fdJac(step, ie, wrt, hist, nxt, prm, out);
#endif
    return;
  }

  out.resize(num_res_, num_next_);

  fem::ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().elem(ie));

  const Index ndof = num_next_;
  Real        s1[kMaxNd * kMaxNd];
  Real        s2[kMaxNd];
  LocalMatrix Ke{ndof, s1};
  LocalVector Fe{ndof, s2};

  HostVector  res_prm;
  const Real* prm_data = fixed_prm_.data();
  if (num_var_prm_ > 0)
  {
    res_prm  = makeResPrm(prm);
    prm_data = res_prm.data();
  }

  if (!assembleLocal(step,
                     vals.numQuadraturePoints(),
                     vals.numNodes(),
                     vals.dim(),
                     num_res_,
                     num_hist_dofs_,
                     num_next_,
                     num_prm_,
                     vals.NData(),
                     vals.dNdxData(),
                     vals.JxWData(),
                     hist.data(),
                     prm_data,
                     Ke,
                     Fe))
  {
    out.setZero();
    return;
  }

  for (Index i = 0; i < num_res_; ++i)
  {
    for (Index j = 0; j < num_next_; ++j)
    {
      out(i, j) = Ke(i, j);
    }
  }
}

#if defined(FEMX_HAS_ENZYME)
void NavierKernel::adJac(Index                  step,
                         Index                  ie,
                         state::VariableBlock   wrt,
                         state::TimeHistoryView hist,
                         const HostVector&      nxt,
                         const HostVector&      prm,
                         DenseMatrix&           out) const
{
  const bool  hist_wrt = wrt.isHistoryState();
  const Index ncol     = hist_wrt ? num_hist_dof_ : num_var_prm_;
  out.resize(num_res_, ncol);
  if (ncol == 0)
  {
    return;
  }

  Index first = 0;
  if (hist_wrt)
  {
    const Index lag = wrt.historyLag();
    if (lag < 0 || lag >= num_hist_)
    {
      throw std::runtime_error(
          "NavierKernel history lag is out of range");
    }
    first = lag * num_hist_dof_;
  }

  fem::ElementValues vals(space_.finiteElement(), quad_);
  vals.reinit(space_.mesh().elem(ie));

  HostVector res_prm = makeResPrm(prm);
  HostVector primal(num_res_);
  HostVector seed(num_res_);
  HostVector hist_adj(num_hist_dofs_);
  HostVector prm_adj(num_prm_);

  for (Index row = 0; row < num_res_; ++row)
  {
    primal.setZero();
    seed.setZero();
    hist_adj.setZero();
    prm_adj.setZero();
    seed[row] = 1.0;

    __enzyme_autodiff<void>(reinterpret_cast<void*>(navierRes),
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
                            num_res_,
                            enzyme_const,
                            num_hist_dofs_,
                            enzyme_const,
                            num_next_,
                            enzyme_const,
                            num_prm_,
                            enzyme_const,
                            vals.NData(),
                            enzyme_const,
                            vals.dNdxData(),
                            enzyme_const,
                            vals.JxWData(),
                            enzyme_dup,
                            hist.data(),
                            hist_adj.data(),
                            enzyme_const,
                            nxt.data(),
                            enzyme_dup,
                            res_prm.data(),
                            prm_adj.data(),
                            enzyme_dup,
                            primal.data(),
                            seed.data());

    const Real* jac_row = hist_wrt ? hist_adj.data() + first
                                   : prm_adj.data();
    for (Index col = 0; col < ncol; ++col)
    {
      out(row, col) = jac_row[col];
    }
  }
}
#endif

void NavierKernel::fdJac(
    Index                  step,
    Index                  ie,
    state::VariableBlock   wrt,
    state::TimeHistoryView hist,
    const HostVector&      nxt,
    const HostVector&      prm,
    DenseMatrix&           out) const
{
  const Index ncol = wrt.isHistoryState()
                         ? num_hist_dof_
                         : num_var_prm_;
  out.resize(num_res_, ncol);
  if (ncol == 0)
  {
    return;
  }

  HostVector hist_vals(num_hist_dofs_);
  for (Index i = 0; i < hist_vals.size(); ++i)
  {
    hist_vals[i] = hist.data()[i];
  }
  HostVector                   prm_vals = prm;
  const state::TimeHistoryView hist_v(
      hist_vals.data(), num_hist_, num_hist_dof_);

  Index hist_first = 0;
  if (wrt.isHistoryState())
  {
    const Index lag = wrt.historyLag();
    if (lag < 0 || lag >= num_hist_)
    {
      throw std::runtime_error(
          "NavierKernel history lag is out of range");
    }
    hist_first = lag * num_hist_dof_;
  }

  for (Index col = 0; col < ncol; ++col)
  {
    Real&      val  = wrt.isHistoryState()
                          ? hist_vals[hist_first + col]
                          : prm_vals[col];
    const Real orig = val;
    const Real eps  = fdStep(orig);

    val = orig + eps;
    HostVector plus;
    res(step, ie, hist_v, nxt, prm_vals, plus);

    val = orig - eps;
    HostVector minus;
    res(step, ie, hist_v, nxt, prm_vals, minus);

    val = orig;
    for (Index row = 0; row < num_res_; ++row)
    {
      out(row, col) = (plus[row] - minus[row]) / (2.0 * eps);
    }
  }
}

void NavierKernel::checkDims()
{
  if (num_res_ < 0 || num_hist_dofs_ < 0
      || num_next_ < 0 || num_var_prm_ < 0 || num_prm_ < 0)
  {
    throw std::runtime_error("NavierKernel received invalid dimensions");
  }
  if (num_next_ <= 0 || num_hist_dofs_ % num_next_ != 0)
  {
    throw std::runtime_error(
        "NavierKernel history size must be a multiple of next-state size");
  }
  num_hist_     = num_hist_dofs_ / num_next_;
  num_hist_dof_ = num_next_;
  if (num_res_ != num_next_ || num_prm_ < 3)
  {
    throw std::runtime_error("NavierKernel received unsupported dimensions");
  }
}

void NavierKernel::checkSizes(state::TimeHistoryView hist,
                              const HostVector&      nxt,
                              const HostVector&      prm) const
{
  if (hist.count() != num_hist_
      || hist.stateSize() != num_hist_dof_
      || nxt.size() != num_next_
      || prm.size() != num_var_prm_)
  {
    throw std::runtime_error("NavierKernel input size mismatch");
  }
}

HostVector NavierKernel::makeResPrm(const HostVector& var_prm) const
{
  if (fixed_prm_.empty())
  {
    return var_prm;
  }

  HostVector out(num_prm_);
  for (Index i = 0; i < num_var_prm_; ++i)
  {
    out[i] = var_prm[i];
  }
  for (Index i = 0; i < fixed_prm_.size(); ++i)
  {
    out[num_var_prm_ + i] = fixed_prm_[i];
  }
  return out;
}

HostVector physicalParams(Real rho, Real mu, Real dt)
{
  if (rho <= 0.0 || mu < 0.0 || dt <= 0.0)
  {
    throw std::runtime_error("NavierKernel received invalid parameters");
  }

  HostVector prm(3);
  prm[0] = rho;
  prm[1] = mu;
  prm[2] = dt;
  return prm;
}

NavierKernel makeNavierKernel(const fem::FESpace&         vel_sp,
                              const fem::GaussQuadrature& quad,
                              Index                       ndof,
                              Real                        rho,
                              Real                        mu,
                              Real                        dt)
{
  if (!supportedSize(quad.size(),
                     vel_sp.numShapesPerElem(),
                     vel_sp.numComponents(),
                     ndof))
  {
    throw std::runtime_error("NavierKernel received unsupported element size");
  }

  return NavierKernel(vel_sp,
                      quad,
                      ndof,
                      2 * ndof,
                      ndof,
                      physicalParams(rho, mu, dt));
}

} // namespace femx::model::ns
