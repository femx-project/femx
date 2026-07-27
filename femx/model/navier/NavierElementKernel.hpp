#pragma once

#include <cmath>

#include <femx/assembly/Assembly.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ElementQuadData.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/navier/FluidProperties.hpp>

namespace femx::linalg
{
class CudaContext;
class CudaSystemMatrix;
} // namespace femx::linalg

namespace femx
{
namespace model
{
namespace navier
{

// The Navier-Stokes model currently accepts Q1 quadrilaterals, P1 triangles,
// and P1 tetrahedra. A tetrahedron therefore has the largest local system:
// four nodes times three velocity components plus pressure.
constexpr Index kMaxNn   = 4;
constexpr Index kMaxNd   = 16;
constexpr Index kMaxNq   = 4;
constexpr Index kMaxDim  = 3;
constexpr Index kNumHist = 2;

namespace detail
{

struct QuadraturePoint
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

FEMX_HOST_DEVICE inline Index vdof(Index in, Index ic, Index dim)
{
  return dim * in + ic;
}

FEMX_HOST_DEVICE inline Index pdof(Index in,
                                   Index num_nodes,
                                   Index dim)
{
  return dim * num_nodes + in;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE Real gradDot(
    const fem::ElementQuadDataView<Space>& data,
    Index                                  ie,
    Index                                  iq,
    Index                                  in,
    Index                                  jn)
{
  Real val = 0.0;
  for (Index id = 0; id < data.dim(); ++id)
  {
    val += data.dNdx(ie, iq, in, id) * data.dNdx(ie, iq, jn, id);
  }
  return val;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE Real advDeriv(
    const fem::ElementQuadDataView<Space>& data,
    const QuadraturePoint&                 qp,
    Index                                  ie,
    Index                                  iq,
    Index                                  in)
{
  Real val = 0.0;
  for (Index id = 0; id < data.dim(); ++id)
  {
    val += data.dNdx(ie, iq, in, id) * qp.adv[id];
  }
  return val;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE Real elemLength(
    const fem::ElementQuadDataView<Space>& data,
    const QuadraturePoint&                 qp,
    Index                                  ie,
    Index                                  iq)
{
  Real speed2 = 0.0;
  for (Index ic = 0; ic < data.dim(); ++ic)
  {
    speed2 += qp.u[ic] * qp.u[ic];
  }

  const Real speed = sqrt(speed2);
  Real       dir[3]{};
  if (speed > 1.0e-10)
  {
    for (Index id = 0; id < data.dim(); ++id)
    {
      dir[id] = qp.u[id] / speed;
    }
  }
  else
  {
    const Real val = 1.0 / sqrt(static_cast<Real>(data.dim()));
    for (Index id = 0; id < data.dim(); ++id)
    {
      dir[id] = val;
    }
  }

  Real sum = 0.0;
  for (Index in = 0; in < data.numShapes(); ++in)
  {
    Real grad = 0.0;
    for (Index id = 0; id < data.dim(); ++id)
    {
      grad += dir[id] * data.dNdx(ie, iq, in, id);
    }
    sum += absVal(grad);
  }
  return sum > 1.0e-14 ? 2.0 / sum : 0.0;
}

template <MemorySpace Space>
FEMX_HOST_DEVICE void evalQp(
    const fem::ElementQuadDataView<Space>&  data,
    const assembly::TimeElementView<Space>& e,
    Index                                   iq,
    FluidProperties                         fluid,
    Real                                    dt,
    QuadraturePoint&                        qp)
{
  for (Index ic = 0; ic < kMaxDim; ++ic)
  {
    qp.u[ic]        = 0.0;
    qp.adv[ic]      = 0.0;
    qp.adv_grad[ic] = 0.0;
    for (Index id = 0; id < kMaxDim; ++id)
    {
      qp.grad[ic][id] = 0.0;
    }
  }
  qp.tau         = 0.0;
  const auto cur = e.histState(0);
  for (Index in = 0; in < data.numShapes(); ++in)
  {
    const Real N = data.N(iq, in);
    for (Index ic = 0; ic < data.dim(); ++ic)
    {
      const Index dof  = vdof(in, ic, data.dim());
      const Real  val  = cur[dof];
      qp.u[ic]        += N * val;
      for (Index id = 0; id < data.dim(); ++id)
      {
        qp.grad[ic][id] += data.dNdx(e.ie, iq, in, id) * val;
      }

      Real adv = val;
      if (e.step > 0 && e.num_hist > 1)
      {
        adv = 1.5 * val - 0.5 * e.histState(1)[dof];
      }
      qp.adv[ic] += N * adv;
    }
  }

  Real speed2 = 0.0;
  for (Index ic = 0; ic < data.dim(); ++ic)
  {
    speed2 += qp.u[ic] * qp.u[ic];
    for (Index id = 0; id < data.dim(); ++id)
    {
      qp.adv_grad[ic] += qp.grad[ic][id] * qp.adv[id];
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

/** @brief Evaluate element rows for CPU and CUDA time assembly. */
template <MemorySpace Space>
class NavierElementKernel
{
public:
  FEMX_HOST_DEVICE NavierElementKernel() = default;

  FEMX_HOST_DEVICE NavierElementKernel(
      fem::ElementQuadDataView<Space> data,
      FluidProperties                 fluid,
      Real                            dt)
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

    if (wrt.isNextState() && !jac.empty())
    {
      res = evalRowImpl(e, row, jac.data());
      return;
    }

    // An empty Jacobian view requests only the residual. History and parameter
    // derivatives are VJPs supplied by Enzyme, so those blocks also discard
    // the next-state Jacobian assembled while evaluating the residual.
    Real lhs[kMaxNd];
    res = evalRowImpl(e, row, lhs);
  }

  /** @brief Return the element quadrature data used by this operator. */
  FEMX_HOST_DEVICE fem::ElementQuadDataView<Space> data() const
  {
    return data_;
  }

  FEMX_HOST_DEVICE FluidProperties fluid() const
  {
    return fluid_;
  }

  FEMX_HOST_DEVICE Real dt() const
  {
    return dt_;
  }

private:
  FEMX_HOST_DEVICE Index numDofs() const
  {
    return (data_.dim() + 1) * data_.numShapes();
  }

  FEMX_HOST_DEVICE Real evalRowImpl(
      const assembly::TimeElementView<Space>& e,
      Index                                   row,
      Real*                                   lhs) const
  {
    const Index num_dofs = numDofs();
    for (Index col = 0; col < num_dofs; ++col)
    {
      lhs[col] = 0.0;
    }

    const Index dim       = data_.dim();
    const Index num_nodes = data_.numShapes();
    const Index num_vel   = dim * num_nodes;
    const bool  vel_row   = row < num_vel;
    const Index in        = vel_row ? row / dim : row - num_vel;
    const Index ic        = vel_row ? row - in * dim : 0;
    Real        rhs       = 0.0;

    for (Index iq = 0; iq < data_.numQuadraturePoints(); ++iq)
    {
      detail::QuadraturePoint qp;
      detail::evalQp(data_, e, iq, fluid_, dt_, qp);
      const Real Jw  = data_.JxW(e.ie, iq);
      const Real Ni  = data_.N(iq, in);
      const Real dvi = detail::advDeriv(data_, qp, e.ie, iq, in);

      if (vel_row)
      {
        // Galerkin transient term.
        rhs += fluid_.rho / dt_ * Ni * qp.u[ic] * Jw;

        // Galerkin advection term.
        rhs -= 0.5 * fluid_.rho * Ni * qp.adv_grad[ic] * Jw;

        // Galerkin diffusion term.
        for (Index id = 0; id < dim; ++id)
        {
          rhs -= 0.5 * fluid_.mu
                 * data_.dNdx(e.ie, iq, in, id) * qp.grad[ic][id] * Jw;
        }

        // SUPG transient term.
        rhs += qp.tau * fluid_.rho / dt_ * dvi * qp.u[ic] * Jw;

        // SUPG advection term.
        rhs -= 0.5 * qp.tau * fluid_.rho * dvi * qp.adv_grad[ic] * Jw;

        for (Index jn = 0; jn < num_nodes; ++jn)
        {
          const Real  Nj  = data_.N(iq, jn);
          const Real  dvj = detail::advDeriv(data_, qp, e.ie, iq, jn);
          const Index ju  = detail::vdof(jn, ic, dim);
          const Index jp  = detail::pdof(jn, num_nodes, dim);

          // Galerkin transient term.
          lhs[ju] += fluid_.rho / dt_ * Ni * Nj * Jw;

          // Galerkin advection term.
          lhs[ju] += 0.5 * fluid_.rho * Ni * dvj * Jw;

          // Galerkin diffusion term.
          lhs[ju] += 0.5 * fluid_.mu
                     * detail::gradDot(data_, e.ie, iq, in, jn) * Jw;

          // SUPG transient term.
          lhs[ju] += qp.tau * fluid_.rho / dt_ * dvi * Nj * Jw;

          // SUPG advection term.
          lhs[ju] += 0.5 * qp.tau * fluid_.rho * dvi * dvj * Jw;

          // Galerkin pressure term.
          lhs[jp] -= data_.dNdx(e.ie, iq, in, ic) * Nj * Jw;

          // SUPG pressure term.
          lhs[jp] += qp.tau * dvi * data_.dNdx(e.ie, iq, jn, ic) * Jw;
        }
      }
      else
      {
        for (Index id = 0; id < dim; ++id)
        {
          // PSPG transient term.
          rhs += qp.tau / dt_
                 * data_.dNdx(e.ie, iq, in, id) * qp.u[id] * Jw;

          // PSPG advection term.
          rhs -= 0.5 * qp.tau
                 * data_.dNdx(e.ie, iq, in, id) * qp.adv_grad[id] * Jw;
        }

        for (Index jn = 0; jn < num_nodes; ++jn)
        {
          const Real Nj  = data_.N(iq, jn);
          const Real dvj = detail::advDeriv(data_, qp, e.ie, iq, jn);
          for (Index id = 0; id < dim; ++id)
          {
            const Index ju = detail::vdof(jn, id, dim);

            // Galerkin continuity term.
            lhs[ju] += Ni * data_.dNdx(e.ie, iq, jn, id) * Jw;

            // PSPG transient term.
            lhs[ju] += qp.tau / dt_
                       * data_.dNdx(e.ie, iq, in, id) * Nj * Jw;

            // PSPG advection term.
            lhs[ju] += 0.5 * qp.tau
                       * data_.dNdx(e.ie, iq, in, id) * dvj * Jw;
          }
          const Index jp = detail::pdof(jn, num_nodes, dim);

          // PSPG pressure term.
          lhs[jp] += qp.tau / fluid_.rho
                     * detail::gradDot(data_, e.ie, iq, in, jn) * Jw;
        }
      }
    }

    rhs = -rhs;
    for (Index col = 0; col < num_dofs; ++col)
    {
      rhs += lhs[col] * e.nxt[col];
    }
    return rhs;
  }

  fem::ElementQuadDataView<Space> data_;
  FluidProperties                 fluid_;
  Real                            dt_{0.0};
};

using HostNavierElementKernel   = NavierElementKernel<MemorySpace::Host>;
using DeviceNavierElementKernel = NavierElementKernel<MemorySpace::Device>;

namespace detail
{

/** @brief Scalar local residual-adjoint product used as an Enzyme VJP root. */
template <MemorySpace Space>
FEMX_HOST_DEVICE Real evalResAdj(Index       num_elems,
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
  const fem::ElementQuadDataView<Space> data{
      num_elems,
      num_qpts,
      num_nodes,
      dim,
      {N, num_qpts * num_nodes},
      {dNdx, num_elems * num_qpts * num_nodes * dim},
      {JxW, num_elems * num_qpts}};

  const assembly::TimeElementView<Space> elem{
      ie,
      step,
      num_hist,
      {hist, num_hist * (data.dim() + 1) * data.numShapes()},
      {nxt, (data.dim() + 1) * data.numShapes()}};

  const NavierElementKernel<Space> kernel(data, {rho, mu}, dt);
  Real                             val = 0.0;

  const Index num_dofs = (data.dim() + 1) * data.numShapes();
  for (Index row = 0; row < num_dofs; ++row)
  {
    Real res = 0.0;
    kernel.evalRow(elem, state::VariableBlock::NextState, row, res, {});
    val += res * adj[row];
  }

  return val;
}

/** @brief One residual row used as the CUDA Enzyme differentiation root. */
template <MemorySpace Space, Index NumQpts, Index NumNodes, Index Dim>
FEMX_HOST_DEVICE Real evalResRowAdj(Index       num_elems,
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
                "Fixed Navier-Stokes dimensions must be positive");

  const fem::ElementQuadDataView<Space> data{
      num_elems,
      NumQpts,
      NumNodes,
      Dim,
      {N, NumQpts * NumNodes},
      {dNdx, num_elems * NumQpts * NumNodes * Dim},
      {JxW, num_elems * NumQpts}};

  const assembly::TimeElementView<Space> elem{
      ie,
      step,
      kNumHist,
      {hist, kNumHist * (data.dim() + 1) * data.numShapes()},
      {nxt, (data.dim() + 1) * data.numShapes()}};

  Real res = 0.0;
  NavierElementKernel<Space>(data, {rho, mu}, dt)
      .evalRow(elem, state::VariableBlock::NextState, row, res, {});

  return res * adj;
}

} // namespace detail

/** @brief Evaluate every Host element history VJP without a matrix. */
void histVjp(const HostNavierElementKernel&       kernel,
             const assembly::HostTimeElementView& e,
             HostVectorView<const Real>           adj,
             HostVectorView<Real>                 out);

namespace detail
{

void assembleNext(
    const DeviceNavierElementKernel&   kernel,
    Index                              step,
    Index                              num_hist,
    Index                              ie_begin,
    Index                              ie_end,
    const assembly::DeviceAssemblyMap& map,
    DeviceVectorView<const Real>       hist,
    DeviceVectorView<const Real>       nxt,
    DeviceVector<Real>&                res,
    linalg::CudaSystemMatrix&          jac,
    linalg::CudaContext&               ctx);

void applyHistJacT(
    const DeviceNavierElementKernel&   kernel,
    Index                              step,
    Index                              num_hist,
    Index                              lag,
    Index                              ie_begin,
    Index                              ie_end,
    const assembly::DeviceAssemblyMap& map,
    DeviceVectorView<const Real>       hist,
    DeviceVectorView<const Real>       nxt,
    DeviceVectorView<const Real>       adj,
    DeviceVector<Real>&                out,
    linalg::CudaContext&               ctx);

} // namespace detail

} // namespace navier
} // namespace model
} // namespace femx
