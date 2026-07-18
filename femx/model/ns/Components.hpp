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

constexpr Index kMaxNn  = 16;
constexpr Index kMaxNd  = 64;
constexpr Index kMaxNq  = 64;
constexpr Index kMaxDim = 3;

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

struct Tangent
{
  Real val{0.0};
  Real der{0.0};

  FEMX_HOST_DEVICE Tangent() = default;

  FEMX_HOST_DEVICE Tangent(Real val_in, Real der_in = 0.0)
    : val(val_in), der(der_in)
  {
  }
};

FEMX_HOST_DEVICE inline Tangent operator+(Tangent lhs, Tangent rhs)
{
  return {lhs.val + rhs.val, lhs.der + rhs.der};
}

FEMX_HOST_DEVICE inline Tangent operator-(Tangent lhs, Tangent rhs)
{
  return {lhs.val - rhs.val, lhs.der - rhs.der};
}

FEMX_HOST_DEVICE inline Tangent operator-(Tangent val)
{
  return {-val.val, -val.der};
}

FEMX_HOST_DEVICE inline Tangent operator*(Tangent lhs, Tangent rhs)
{
  return {lhs.val * rhs.val,
          lhs.der * rhs.val + lhs.val * rhs.der};
}

FEMX_HOST_DEVICE inline Tangent operator/(Tangent lhs, Tangent rhs)
{
  const Real den = rhs.val * rhs.val;
  return {lhs.val / rhs.val,
          (lhs.der * rhs.val - lhs.val * rhs.der) / den};
}

FEMX_HOST_DEVICE inline Tangent& operator+=(Tangent& lhs, Tangent rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

FEMX_HOST_DEVICE inline Real primal(Real val)
{
  return val;
}

FEMX_HOST_DEVICE inline Real primal(Tangent val)
{
  return val.val;
}

FEMX_HOST_DEVICE inline Real root(Real val)
{
  return sqrt(val);
}

FEMX_HOST_DEVICE inline Tangent root(Tangent val)
{
  const Real out = sqrt(val.val);
  return {out, out > 0.0 ? val.der / (2.0 * out) : 0.0};
}

template <class Scalar>
struct NavierQp
{
  Scalar u[3]{};
  Scalar adv[3]{};
  Scalar grad[3][3]{};
  Scalar adv_grad[3]{};
  Scalar tau{};
};

FEMX_HOST_DEVICE inline Real absVal(Real val)
{
  return val >= 0.0 ? val : -val;
}

FEMX_HOST_DEVICE inline Tangent absVal(Tangent val)
{
  return val.val >= 0.0 ? val : -val;
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

template <MemorySpace Space, class Scalar>
FEMX_HOST_DEVICE Scalar advDeriv(const NavierDataView<Space>& data,
                                 const NavierQp<Scalar>&      qp,
                                 Index                        ie,
                                 Index                        iq,
                                 Index                        in)
{
  Scalar val{};
  for (Index d = 0; d < data.dim(); ++d)
  {
    val += data.dNdx(ie, iq, in, d) * qp.adv[d];
  }
  return val;
}

template <MemorySpace Space, class Scalar>
FEMX_HOST_DEVICE Scalar elemLength(const NavierDataView<Space>& data,
                                   const NavierQp<Scalar>&      qp,
                                   Index                        ie,
                                   Index                        iq)
{
  Scalar speed2{};
  for (Index d = 0; d < data.dim(); ++d)
  {
    speed2 += qp.u[d] * qp.u[d];
  }

  const Scalar speed = root(speed2);
  Scalar       dir[3]{};
  if (primal(speed) > 1.0e-10)
  {
    for (Index d = 0; d < data.dim(); ++d)
    {
      dir[d] = qp.u[d] / speed;
    }
  }
  else
  {
    const Scalar val = 1.0 / sqrt(static_cast<Real>(data.dim()));
    for (Index d = 0; d < data.dim(); ++d)
    {
      dir[d] = val;
    }
  }

  Scalar sum{};
  for (Index in = 0; in < data.numNodes(); ++in)
  {
    Scalar grad{};
    for (Index d = 0; d < data.dim(); ++d)
    {
      grad += dir[d] * data.dNdx(ie, iq, in, d);
    }
    sum += absVal(grad);
  }
  return primal(sum) > 1.0e-14 ? 2.0 / sum : Scalar{};
}

template <class Scalar>
FEMX_HOST_DEVICE Scalar seeded(Real  val,
                               Index lag,
                               Index id,
                               Index seed_lag,
                               Index seed_col)
{
  (void) lag;
  (void) id;
  (void) seed_lag;
  (void) seed_col;
  return Scalar(val);
}

template <>
FEMX_HOST_DEVICE inline Tangent seeded<Tangent>(Real  val,
                                                Index lag,
                                                Index id,
                                                Index seed_lag,
                                                Index seed_col)
{
  return {val, lag == seed_lag && id == seed_col ? 1.0 : 0.0};
}

template <MemorySpace Space, class Scalar>
FEMX_HOST_DEVICE NavierQp<Scalar> evalQp(
    const NavierDataView<Space>&            data,
    const assembly::TimeElementView<Space>& e,
    Index                                   iq,
    KernelFluid                             fluid,
    Real                                    dt,
    Index                                   seed_lag,
    Index                                   seed_col)
{
  NavierQp<Scalar> qp{};
  const auto       cur = e.histState(0);
  for (Index in = 0; in < data.numNodes(); ++in)
  {
    const Real N = data.N(iq, in);
    for (Index c = 0; c < data.dim(); ++c)
    {
      const Index  id   = vdof(in, c, data.dim());
      const Scalar val  = seeded<Scalar>(cur[id], 0, id, seed_lag, seed_col);
      qp.u[c]          += N * val;
      for (Index d = 0; d < data.dim(); ++d)
      {
        qp.grad[c][d] += data.dNdx(e.ie, iq, in, d) * val;
      }

      Scalar adv = val;
      if (e.step > 0 && e.num_hist > 1)
      {
        const Scalar old = seeded<Scalar>(
            e.histState(1)[id], 1, id, seed_lag, seed_col);
        adv = 1.5 * val - 0.5 * old;
      }
      qp.adv[c] += N * adv;
    }
  }

  Scalar speed2{};
  for (Index c = 0; c < data.dim(); ++c)
  {
    speed2 += qp.u[c] * qp.u[c];
    for (Index d = 0; d < data.dim(); ++d)
    {
      qp.adv_grad[c] += qp.grad[c][d] * qp.adv[d];
    }
  }

  const Scalar speed = root(speed2);
  const Scalar h     = elemLength(data, qp, e.ie, iq);
  const Real   nu    = fluid.mu / fluid.rho;
  const Real   time  = (2.0 / dt) * (2.0 / dt);
  Scalar       flow{};
  Scalar       diff{};
  if (primal(h) > 0.0)
  {
    flow = (2.0 * speed / h) * (2.0 * speed / h);
    diff = (4.0 * nu / (h * h)) * (4.0 * nu / (h * h));
  }
  qp.tau = 1.0 / root(Scalar(time) + flow + diff);
  return qp;
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
      res = evalPhysicsRow<Real>(e, row, -1, -1, jac.data());
      return;
    }

    if (wrt.isHistoryState() && wrt.historyLag() >= 0
        && wrt.historyLag() < e.num_hist)
    {
      detail::Tangent K[kMaxNd];
      for (Index col = 0; col < jac.size(); ++col)
      {
        const detail::Tangent out = evalPhysicsRow<detail::Tangent>(
            e, row, wrt.historyLag(), col, K);
        if (col == 0)
        {
          res = out.val;
        }
        jac[col] = out.der;
      }
      if (jac.empty())
      {
        Real K_real[kMaxNd];
        res = evalPhysicsRow<Real>(e, row, -1, -1, K_real);
      }
      return;
    }

    // The built-in operator has no element parameter block. Invalid history
    // blocks likewise receive a zero tangent while retaining the true row.
    Real K[kMaxNd];
    res = evalPhysicsRow<Real>(e, row, -1, -1, K);
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
  template <class Scalar>
  FEMX_HOST_DEVICE Scalar evalPhysicsRow(
      const assembly::TimeElementView<Space>& e,
      Index                                   row,
      Index                                   seed_lag,
      Index                                   seed_col,
      Scalar*                                 K) const
  {
    const Index num_dofs = data_.numDofs();
    for (Index col = 0; col < num_dofs; ++col)
    {
      K[col] = Scalar{};
    }

    const Index dim       = data_.dim();
    const Index num_nodes = data_.numNodes();
    const Index num_vel   = dim * num_nodes;
    const bool  vel_row   = row < num_vel;
    const Index i         = vel_row ? row / dim : row - num_vel;
    const Index c         = vel_row ? row - i * dim : 0;
    Scalar      F{};

    for (Index iq = 0; iq < data_.numQpts(); ++iq)
    {
      const detail::NavierQp<Scalar> qp = detail::evalQp<Space, Scalar>(
          data_, e, iq, fluid_, dt_, seed_lag, seed_col);
      const Real   Jw  = data_.JxW(e.ie, iq);
      const Real   Ni  = data_.N(iq, i);
      const Scalar dvi = detail::advDeriv(data_, qp, e.ie, iq, i);

      if (vel_row)
      {
        Scalar grad_u{};
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
          const Real   Nj  = data_.N(iq, j);
          const Scalar dvj = detail::advDeriv(data_, qp, e.ie, iq, j);
          const Index  ju  = detail::vdof(j, c, dim);
          const Index  jp  = detail::pdof(j, num_nodes, dim);

          K[ju] +=
              (fluid_.rho / dt_ * Ni * Nj
               + 0.5 * fluid_.rho * Ni * dvj
               + 0.5 * fluid_.mu
                     * detail::gradDot(data_, e.ie, iq, i, j)
               + qp.tau * fluid_.rho / dt_ * dvi * Nj
               + 0.5 * qp.tau * fluid_.rho * dvi * dvj)
              * Jw;
          K[jp] +=
              (-data_.dNdx(e.ie, iq, i, c) * Nj
               + qp.tau * dvi * data_.dNdx(e.ie, iq, j, c))
              * Jw;
        }
      }
      else
      {
        Scalar div_u{};
        Scalar div_adv{};
        for (Index d = 0; d < dim; ++d)
        {
          const Real grad  = data_.dNdx(e.ie, iq, i, d);
          div_u           += grad * qp.u[d];
          div_adv         += grad * qp.adv_grad[d];
        }
        F += (qp.tau / dt_ * div_u - 0.5 * qp.tau * div_adv) * Jw;

        for (Index j = 0; j < num_nodes; ++j)
        {
          const Real   Nj  = data_.N(iq, j);
          const Scalar dvj = detail::advDeriv(data_, qp, e.ie, iq, j);
          for (Index d = 0; d < dim; ++d)
          {
            const Index ju = detail::vdof(j, d, dim);
            const Real  gi = data_.dNdx(e.ie, iq, i, d);
            K[ju] +=
                (Ni * data_.dNdx(e.ie, iq, j, d)
                 + qp.tau / dt_ * gi * Nj
                 + 0.5 * qp.tau * gi * dvj)
                * Jw;
          }
          const Index jp  = detail::pdof(j, num_nodes, dim);
          K[jp]          += qp.tau / fluid_.rho
                   * detail::gradDot(data_, e.ie, iq, i, j) * Jw;
        }
      }
    }

    Scalar out = -F;
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

} // namespace ns
} // namespace model
} // namespace femx
