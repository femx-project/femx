#pragma once

#include <femx/common/Checks.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/state/Residual.hpp>

namespace femx::tests::stationary
{

inline void setScalar(HostCsrMatrix& mat, Real val)
{
  require(mat.rows() == 1 && mat.cols() == 1 && mat.nnz() == 1,
          "Scalar Host matrix has an invalid graph");
  mat.valsData()[0] = val;
}

template <class Backend>
class AffineResidual final : public state::Residual<Backend>
{
  static_assert(Backend::space == MemorySpace::Host,
                "Affine test residual requires Host storage");

public:
  using Vec   = typename Backend::Vec;
  using Mat   = typename Backend::Mat;
  using Graph = typename Backend::Graph;
  using Ctx   = typename Backend::Ctx;

  explicit AffineResidual(Real coeff)
    : coeff_(coeff),
      graph_(1, 1, HostIndexVector{0, 1}, HostIndexVector{0})
  {
    require(coeff_ != 0.0, "Affine residual coefficient must be nonzero");
  }

  state::Dimensions dims() const override
  {
    return {1, 1, 1};
  }

  const HostCsrGraph& hostGraph() const override
  {
    return graph_;
  }

  const Graph& graph() const override
  {
    return graph_;
  }

  void res(const Vec& state,
           const Vec& prm,
           Vec&       out,
           Ctx&) const override
  {
    check(state, prm);
    out.assign(1, coeff_ * state[0] - prm[0]);
  }

  void assembleStateJac(const Vec& state,
                        const Vec& prm,
                        Mat&       out,
                        Ctx&) const override
  {
    check(state, prm);
    setScalar(out, coeff_);
  }

  void applyParamJacT(const Vec& state,
                      const Vec& prm,
                      const Vec& adj,
                      Vec&       out,
                      Ctx&) const override
  {
    check(state, prm);
    require(adj.size() == 1, "Affine residual adjoint size mismatch");
    out.assign(1, -adj[0]);
  }

private:
  static void check(const Vec& state, const Vec& prm)
  {
    require(state.size() == 1 && prm.size() == 1,
            "Affine residual vector size mismatch");
  }

  Real         coeff_{1.0};
  HostCsrGraph graph_;
};

class QuadraticObjective final : public inverse::Objective
{
public:
  QuadraticObjective(Real target, Real beta)
    : target_(target), beta_(beta)
  {
  }

  Index numStates() const override
  {
    return 1;
  }

  Index numParams() const override
  {
    return 1;
  }

  Real value(const HostVector& state,
             const HostVector& prm) const override
  {
    const Real diff = state[0] - target_;
    return 0.5 * diff * diff + 0.5 * beta_ * prm[0] * prm[0];
  }

  void stateGrad(const HostVector& state,
                 const HostVector&,
                 HostVector& out) const override
  {
    out.assign(1, state[0] - target_);
  }

  void paramGrad(const HostVector&,
                 const HostVector& prm,
                 HostVector&       out) const override
  {
    out.assign(1, beta_ * prm[0]);
  }

private:
  Real target_{0.0};
  Real beta_{0.0};
};

class QuadraticResidual final
  : public state::Residual<linalg::HostCsrBackend>
{
public:
  QuadraticResidual()
    : graph_(1, 1, HostIndexVector{0, 1}, HostIndexVector{0})
  {
  }

  state::Dimensions dims() const override
  {
    return {1, 1, 1};
  }

  const HostCsrGraph& hostGraph() const override
  {
    return graph_;
  }

  const HostCsrGraph& graph() const override
  {
    return graph_;
  }

  void res(const HostVector& state,
           const HostVector& prm,
           HostVector&       out,
           CpuContext&) const override
  {
    out.assign(1, state[0] * state[0] - prm[0]);
  }

  void assembleStateJac(const HostVector& state,
                        const HostVector&,
                        HostCsrMatrix& out,
                        CpuContext&) const override
  {
    setScalar(out, 2.0 * state[0]);
  }

  void applyParamJacT(const HostVector&,
                      const HostVector&,
                      const HostVector& adj,
                      HostVector&       out,
                      CpuContext&) const override
  {
    out.assign(1, -adj[0]);
  }

private:
  HostCsrGraph graph_;
};

} // namespace femx::tests::stationary
