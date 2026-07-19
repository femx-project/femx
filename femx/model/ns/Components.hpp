#pragma once

#include <cmath>

#include <femx/assembly/Assembly.hpp>
#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{
class FESpace;
class GaussQuadrature;
} // namespace fem

namespace model
{
namespace ns
{

// NavierStokesModel currently accepts Q1 quadrilaterals, P1 triangles, and P1
// tetrahedra. A tetrahedron therefore has the largest local system: four
// nodes times three velocity components plus pressure.
constexpr Index kMaxNn  = 4;
constexpr Index kMaxNd  = 16;
constexpr Index kMaxNq  = 4;
constexpr Index kMaxDim = 3;
constexpr Index kNumHist = 2;

template <MemorySpace Space>
class NavierData;

using HostNavierData   = NavierData<MemorySpace::Host>;
using DeviceNavierData = NavierData<MemorySpace::Device>;

/** @brief Flatten all element values used by the Navier operator. */
HostNavierData makeNavierData(const fem::FESpace&         vel_sp,
                              const fem::GaussQuadrature& quad);

/** @brief Copy reusable Navier element data to Device storage. */
void copy(const HostNavierData& src,
          DeviceNavierData&     dst,
          CudaContext&          ctx);

/** @brief Non-owning, trivially-copyable Navier element-data view. */
template <MemorySpace Space>
class NavierDataView
{
public:
  FEMX_HOST_DEVICE NavierDataView() = default;

  FEMX_HOST_DEVICE NavierDataView(
      Index                         num_elems,
      Index                         num_qpts,
      Index                         num_nodes,
      Index                         dim,
      VectorView<Space, const Real> N,
      VectorView<Space, const Real> dNdx,
      VectorView<Space, const Real> JxW)
    : num_elems_(num_elems),
      num_qpts_(num_qpts),
      num_nodes_(num_nodes),
      dim_(dim),
      N_(N),
      dNdx_(dNdx),
      JxW_(JxW)
  {
  }

  FEMX_HOST_DEVICE Index numElems() const
  {
    return num_elems_;
  }

  FEMX_HOST_DEVICE Index numQpts() const
  {
    return num_qpts_;
  }

  FEMX_HOST_DEVICE Index numNodes() const
  {
    return num_nodes_;
  }

  FEMX_HOST_DEVICE Index dim() const
  {
    return dim_;
  }

  FEMX_HOST_DEVICE Index numDofs() const
  {
    return (dim_ + 1) * num_nodes_;
  }

  FEMX_HOST_DEVICE Real N(Index iq, Index in) const
  {
    return N_[iq * num_nodes_ + in];
  }

  FEMX_HOST_DEVICE Real dNdx(Index ie,
                             Index iq,
                             Index in,
                             Index d) const
  {
    return dNdx_[((ie * num_qpts_ + iq) * num_nodes_ + in) * dim_ + d];
  }

  FEMX_HOST_DEVICE Real JxW(Index ie, Index iq) const
  {
    return JxW_[ie * num_qpts_ + iq];
  }

  FEMX_HOST_DEVICE const Real* NData() const
  {
    return N_.data();
  }

  FEMX_HOST_DEVICE const Real* dNdxData() const
  {
    return dNdx_.data();
  }

  FEMX_HOST_DEVICE const Real* JxWData() const
  {
    return JxW_.data();
  }

private:
  Index                         num_elems_{0};
  Index                         num_qpts_{0};
  Index                         num_nodes_{0};
  Index                         dim_{0};
  VectorView<Space, const Real> N_;
  VectorView<Space, const Real> dNdx_;
  VectorView<Space, const Real> JxW_;
};

/** @brief Memory-space owner for reusable Navier quadrature data. */
template <MemorySpace Space>
class NavierData
{
public:
  NavierData() = default;

  NavierData(const NavierData&)                = default;
  NavierData(NavierData&&) noexcept            = default;
  NavierData& operator=(const NavierData&)     = default;
  NavierData& operator=(NavierData&&) noexcept = default;

  Index numElems() const noexcept
  {
    return num_elems_;
  }

  Index numQpts() const noexcept
  {
    return num_qpts_;
  }

  Index numNodes() const noexcept
  {
    return num_nodes_;
  }

  Index dim() const noexcept
  {
    return dim_;
  }

  Index numDofs() const noexcept
  {
    return (dim_ + 1) * num_nodes_;
  }

  NavierDataView<Space> view() const noexcept
  {
    return {num_elems_,
            num_qpts_,
            num_nodes_,
            dim_,
            N_.view(),
            dNdx_.view(),
            JxW_.view()};
  }

private:
  friend HostNavierData makeNavierData(
      const fem::FESpace&         vel_sp,
      const fem::GaussQuadrature& quad);

  friend void copy(const HostNavierData& src,
                   DeviceNavierData&     dst,
                   CudaContext&          ctx);

  Index               num_elems_{0};
  Index               num_qpts_{0};
  Index               num_nodes_{0};
  Index               dim_{0};
  Vector<Space, Real> N_;
  Vector<Space, Real> dNdx_;
  Vector<Space, Real> JxW_;
};

inline void copy(const HostNavierData& src,
                 DeviceNavierData&     dst,
                 CudaContext&          ctx)
{
  dst.num_elems_ = src.num_elems_;
  dst.num_qpts_  = src.num_qpts_;
  dst.num_nodes_ = src.num_nodes_;
  dst.dim_       = src.dim_;
  femx::copy(src.N_, dst.N_, ctx);
  femx::copy(src.dNdx_, dst.dNdx_, ctx);
  femx::copy(src.JxW_, dst.JxW_, ctx);
}

struct KernelFluid
{
  Real rho{1.0};
  Real mu{1.0};
};

namespace detail
{

struct NavierQp
{
  Real u[3];
  Real adv[3];
  Real grad[3][3];
  Real adv_grad[3];
  Real tau;
};

FEMX_HOST_DEVICE inline Real absVal(Real val)
{
  return val >= 0.0 ? val : -val;
}

FEMX_HOST_DEVICE inline Index vdof(Index in, Index comp, Index dim)
{
  return dim * in + comp;
}

FEMX_HOST_DEVICE inline Index pdof(Index in,
                                   Index num_nodes,
                                   Index dim)
{
  return dim * num_nodes + in;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE Real gradDot(const NavierDataView<Space>& data,
                              Index                        ie,
                              Index                        iq,
                              Index                        i,
                              Index                        j)
{
  Real val = 0.0;
  for (Index d = 0; d < data.dim(); ++d)
  {
    val += data.dNdx(ie, iq, i, d) * data.dNdx(ie, iq, j, d);
  }
  return val;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE Real advDeriv(const NavierDataView<Space>& data,
                               const NavierQp&              qp,
                               Index                        ie,
                               Index                        iq,
                               Index                        in)
{
  Real val = 0.0;
  for (Index d = 0; d < data.dim(); ++d)
  {
    val += data.dNdx(ie, iq, in, d) * qp.adv[d];
  }
  return val;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE Real elemLength(const NavierDataView<Space>& data,
                                 const NavierQp&              qp,
                                 Index                        ie,
                                 Index                        iq)
{
  Real speed2 = 0.0;
  for (Index d = 0; d < data.dim(); ++d)
  {
    speed2 += qp.u[d] * qp.u[d];
  }

  const Real speed = sqrt(speed2);
  Real       dir[3]{};
  if (speed > 1.0e-10)
  {
    for (Index d = 0; d < data.dim(); ++d)
    {
      dir[d] = qp.u[d] / speed;
    }
  }
  else
  {
    const Real val = 1.0 / sqrt(static_cast<Real>(data.dim()));
    for (Index d = 0; d < data.dim(); ++d)
    {
      dir[d] = val;
    }
  }

  Real sum = 0.0;
  for (Index in = 0; in < data.numNodes(); ++in)
  {
    Real grad = 0.0;
    for (Index d = 0; d < data.dim(); ++d)
    {
      grad += dir[d] * data.dNdx(ie, iq, in, d);
    }
    sum += absVal(grad);
  }
  return sum > 1.0e-14 ? 2.0 / sum : 0.0;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE void evalQp(
    const NavierDataView<Space>&            data,
    const assembly::TimeElementView<Space>& e,
    Index                                   iq,
    KernelFluid                             fluid,
    Real                                    dt,
    NavierQp&                               qp)
{
  for (Index c = 0; c < kMaxDim; ++c)
  {
    qp.u[c]        = 0.0;
    qp.adv[c]      = 0.0;
    qp.adv_grad[c] = 0.0;
    for (Index d = 0; d < kMaxDim; ++d)
    {
      qp.grad[c][d] = 0.0;
    }
  }
  qp.tau         = 0.0;
  const auto cur = e.histState(0);
  for (Index in = 0; in < data.numNodes(); ++in)
  {
    const Real N = data.N(iq, in);
    for (Index c = 0; c < data.dim(); ++c)
    {
      const Index id   = vdof(in, c, data.dim());
      const Real  val  = cur[id];
      qp.u[c]         += N * val;
      for (Index d = 0; d < data.dim(); ++d)
      {
        qp.grad[c][d] += data.dNdx(e.ie, iq, in, d) * val;
      }

      Real adv = val;
      if (e.step > 0 && e.num_hist > 1)
      {
        adv = 1.5 * val - 0.5 * e.histState(1)[id];
      }
      qp.adv[c] += N * adv;
    }
  }

  Real speed2 = 0.0;
  for (Index c = 0; c < data.dim(); ++c)
  {
    speed2 += qp.u[c] * qp.u[c];
    for (Index d = 0; d < data.dim(); ++d)
    {
      qp.adv_grad[c] += qp.grad[c][d] * qp.adv[d];
    }
  }

  const Real speed = sqrt(speed2);
  const Real h     = elemLength(data, qp, e.ie, iq);
  const Real nu    = fluid.mu / fluid.rho;
  const Real time  = (2.0 / dt) * (2.0 / dt);
  Real       flow  = 0.0;
  Real       diff  = 0.0;
  if (h > 0.0)
  {
    flow = (2.0 * speed / h) * (2.0 * speed / h);
    diff = (4.0 * nu / (h * h)) * (4.0 * nu / (h * h));
  }
  qp.tau = 1.0 / sqrt(time + flow + diff);
}

} // namespace detail

/** @brief Row-wise Navier operator shared by CPU and CUDA time assembly. */
template <MemorySpace Space>
class NavierOperator
{
public:
  FEMX_HOST_DEVICE NavierOperator() = default;

  FEMX_HOST_DEVICE NavierOperator(NavierDataView<Space> data,
                                  KernelFluid           fluid,
                                  Real                  dt)
    : data_(data), fluid_(fluid), dt_(dt)
  {
  }

  FEMX_HOST_DEVICE void evalRow(
      const assembly::TimeElementView<Space>& e,
      state::VariableBlock                    wrt,
      Index                                   row,
      Real&                                   res,
      VectorView<Space, Real>                 jac) const
  {
    res = 0.0;
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = 0.0;
    }

    if (wrt.isNextState())
    {
      res = evalPhysicsRow(e, row, jac.data());
      return;
    }

    // History and parameter derivatives are VJPs supplied by Enzyme. The row
    // operator remains next-state-only so no history CSR path can reappear.
    Real K[kMaxNd];
    res = evalPhysicsRow(e, row, K);
  }

  FEMX_HOST_DEVICE void evalResidual(
      const assembly::TimeElementView<Space>& e,
      Real*                                   res) const
  {
    Real K[kMaxNd];
    for (Index row = 0; row < data_.numDofs(); ++row)
    {
      res[row] = evalPhysicsRow(e, row, K);
    }
  }

  /** @brief Evaluate one element residual row without materializing a vector. */
  FEMX_HOST_DEVICE Real evalResidualRow(
      const assembly::TimeElementView<Space>& e,
      Index                                   row) const
  {
    Real K[kMaxNd];
    return evalPhysicsRow(e, row, K);
  }

  FEMX_HOST_DEVICE NavierDataView<Space> data() const
  {
    return data_;
  }

  FEMX_HOST_DEVICE KernelFluid fluid() const
  {
    return fluid_;
  }

  FEMX_HOST_DEVICE Real dt() const
  {
    return dt_;
  }

private:
  FEMX_HOST_DEVICE Real evalPhysicsRow(
      const assembly::TimeElementView<Space>& e,
      Index                                   row,
      Real*                                   K) const
  {
    const Index num_dofs = data_.numDofs();
    for (Index col = 0; col < num_dofs; ++col)
    {
      K[col] = 0.0;
    }

    const Index dim       = data_.dim();
    const Index num_nodes = data_.numNodes();
    const Index num_vel   = dim * num_nodes;
    const bool  vel_row   = row < num_vel;
    const Index i         = vel_row ? row / dim : row - num_vel;
    const Index c         = vel_row ? row - i * dim : 0;
    Real        F         = 0.0;

    for (Index iq = 0; iq < data_.numQpts(); ++iq)
    {
      detail::NavierQp qp;
      detail::evalQp(data_, e, iq, fluid_, dt_, qp);
      const Real Jw  = data_.JxW(e.ie, iq);
      const Real Ni  = data_.N(iq, i);
      const Real dvi = detail::advDeriv(data_, qp, e.ie, iq, i);

      if (vel_row)
      {
        Real grad_u = 0.0;
        for (Index d = 0; d < dim; ++d)
        {
          grad_u += data_.dNdx(e.ie, iq, i, d) * qp.grad[c][d];
        }
        F += (fluid_.rho / dt_ * Ni * qp.u[c]
              - 0.5 * fluid_.rho * Ni * qp.adv_grad[c]
              - 0.5 * fluid_.mu * grad_u
              + qp.tau * fluid_.rho / dt_ * dvi * qp.u[c]
              - 0.5 * qp.tau * fluid_.rho * dvi * qp.adv_grad[c])
             * Jw;

        for (Index j = 0; j < num_nodes; ++j)
        {
          const Real  Nj  = data_.N(iq, j);
          const Real  dvj = detail::advDeriv(data_, qp, e.ie, iq, j);
          const Index ju  = detail::vdof(j, c, dim);
          const Index jp  = detail::pdof(j, num_nodes, dim);

          K[ju] += (fluid_.rho / dt_ * Ni * Nj
                    + 0.5 * fluid_.rho * Ni * dvj
                    + 0.5 * fluid_.mu * detail::gradDot(data_, e.ie, iq, i, j)
                    + qp.tau * fluid_.rho / dt_ * dvi * Nj
                    + 0.5 * qp.tau * fluid_.rho * dvi * dvj)
                   * Jw;
          K[jp] += (-data_.dNdx(e.ie, iq, i, c) * Nj
                    + qp.tau * dvi * data_.dNdx(e.ie, iq, j, c))
                   * Jw;
        }
      }
      else
      {
        Real div_u   = 0.0;
        Real div_adv = 0.0;
        for (Index d = 0; d < dim; ++d)
        {
          const Real grad  = data_.dNdx(e.ie, iq, i, d);
          div_u           += grad * qp.u[d];
          div_adv         += grad * qp.adv_grad[d];
        }
        F += (qp.tau / dt_ * div_u - 0.5 * qp.tau * div_adv) * Jw;

        for (Index j = 0; j < num_nodes; ++j)
        {
          const Real Nj  = data_.N(iq, j);
          const Real dvj = detail::advDeriv(data_, qp, e.ie, iq, j);
          for (Index d = 0; d < dim; ++d)
          {
            const Index ju = detail::vdof(j, d, dim);
            const Real  gi = data_.dNdx(e.ie, iq, i, d);

            K[ju] += (Ni * data_.dNdx(e.ie, iq, j, d)
                      + qp.tau / dt_ * gi * Nj
                      + 0.5 * qp.tau * gi * dvj)
                     * Jw;
          }
          const Index jp = detail::pdof(j, num_nodes, dim);

          K[jp] += qp.tau / fluid_.rho
                   * detail::gradDot(data_, e.ie, iq, i, j) * Jw;
        }
      }
    }

    Real out = -F;
    for (Index col = 0; col < num_dofs; ++col)
    {
      out += K[col] * e.nxt[col];
    }
    return out;
  }

  NavierDataView<Space> data_;
  KernelFluid           fluid_;
  Real                  dt_{0.0};
};

namespace detail
{

/** @brief Shared local NS residual differentiated by Host and CUDA Enzyme. */
template <MemorySpace Space>
FEMX_HOST_DEVICE void evalNavierRes(NavierDataView<Space> data,
                                    KernelFluid           fluid,
                                    Real                  dt,
                                    Index                 ie,
                                    Index                 step,
                                    Index                 num_hist,
                                    const Real*           hist,
                                    const Real*           nxt,
                                    Real*                 res)
{
  const Index                            ncol = data.numDofs();
  const assembly::TimeElementView<Space> e{
      ie,
      step,
      num_hist,
      {hist, num_hist * ncol},
      {nxt, ncol}};
  NavierOperator<Space>(data, fluid, dt).evalResidual(e, res);
}

/** @brief Scalar local residual-adjoint product used as an Enzyme VJP root. */
template <MemorySpace Space>
FEMX_HOST_DEVICE Real evalNavierAdj(Index       num_elems,
                                    Index       num_qpts,
                                    Index       num_nodes,
                                    Index       dim,
                                    const Real* N,
                                    const Real* dNdx,
                                    const Real* JxW,
                                    Real        rho,
                                    Real        mu,
                                    Real        dt,
                                    Index       ie,
                                    Index       step,
                                    Index       num_hist,
                                    const Real* hist,
                                    const Real* nxt,
                                    const Real* adj)
{
  const NavierDataView<Space> data{
      num_elems,
      num_qpts,
      num_nodes,
      dim,
      {N, num_qpts * num_nodes},
      {dNdx, num_elems * num_qpts * num_nodes * dim},
      {JxW, num_elems * num_qpts}};
  const assembly::TimeElementView<Space> e{
      ie,
      step,
      num_hist,
      {hist, num_hist * data.numDofs()},
      {nxt, data.numDofs()}};
  const NavierOperator<Space> op(data, {rho, mu}, dt);
  Real                        val = 0.0;
  for (Index row = 0; row < data.numDofs(); ++row)
  {
    val += op.evalResidualRow(e, row) * adj[row];
  }
  return val;
}

/** @brief One residual row used as the CUDA Enzyme differentiation root. */
template <MemorySpace Space, Index NumQpts, Index NumNodes, Index Dim>
FEMX_HOST_DEVICE Real evalNavierRowAdj(Index       num_elems,
                                       const Real* N,
                                       const Real* dNdx,
                                       const Real* JxW,
                                       Real        rho,
                                       Real        mu,
                                       Real        dt,
                                       Index       ie,
                                       Index       row,
                                       Index       step,
                                       const Real* hist,
                                       const Real* nxt,
                                       Real        adj)
{
  static_assert(NumQpts > 0 && NumNodes > 0 && Dim > 0,
                "Fixed Navier dimensions must be positive");
  const NavierDataView<Space> data{
      num_elems,
      NumQpts,
      NumNodes,
      Dim,
      {N, NumQpts * NumNodes},
      {dNdx, num_elems * NumQpts * NumNodes * Dim},
      {JxW, num_elems * NumQpts}};
  const assembly::TimeElementView<Space> e{
      ie,
      step,
      kNumHist,
      {hist, kNumHist * data.numDofs()},
      {nxt, data.numDofs()}};
  return NavierOperator<Space>(data, {rho, mu}, dt).evalResidualRow(e, row)
         * adj;
}

} // namespace detail

/** @brief Evaluate every Host element history VJP without a matrix. */
void histVjp(const NavierOperator<MemorySpace::Host>&            op,
             const assembly::TimeElementView<MemorySpace::Host>& e,
             HostConstVectorView                                 adj,
             HostVectorView                                      out);

namespace detail
{

/** @brief CUDA launchers used by the backend-neutral NS residual. */
void evalNavierRes(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      num_hist,
    Index                                      ie_begin,
    Index                                      ie_end,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceVector&                              out,
    CudaContext&                               ctx);

void assembleNavierNext(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      num_hist,
    Index                                      ie_begin,
    Index                                      ie_end,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceVector&                              res,
    DeviceCsrMatrix&                           jac,
    CudaContext&                               ctx);

void applyNavierHistJacT(
    const NavierOperator<MemorySpace::Device>& op,
    Index                                      step,
    Index                                      num_hist,
    Index                                      lag,
    Index                                      ie_begin,
    Index                                      ie_end,
    const assembly::DeviceAssemblyMap&         map,
    DeviceConstVectorView                      hist,
    DeviceConstVectorView                      nxt,
    DeviceConstVectorView                      adj,
    DeviceVector&                              out,
    CudaContext&                               ctx);

} // namespace detail

} // namespace ns
} // namespace model
} // namespace femx
