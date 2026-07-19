#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Backend.hpp>

namespace femx::state
{

/** @brief Sizes for a parameter-dependent stationary residual. */
struct Dimensions
{
  Index num_states{0};
  Index num_param{0};
  Index num_res{0};
};

/** @brief Stationary residual contract over one concrete backend. */
template <class Backend>
class Residual
{
  static_assert(linalg::is_backend_v<Backend>,
                "Residual requires a valid backend type");

public:
  static constexpr MemorySpace space = Backend::space;

  using Vec   = typename Backend::Vec;
  using Mat   = typename Backend::Mat;
  using Graph = typename Backend::Graph;
  using Ctx   = typename Backend::Ctx;

  virtual ~Residual() = default;

  virtual Dimensions dims() const = 0;

  /** @brief Return the Host graph used for metadata and backend conversion. */
  virtual const HostCsrGraph& hostGraph() const = 0;

  /** @brief Return the state-Jacobian graph in backend storage. */
  virtual const Graph& graph() const = 0;

  /** @brief Evaluate R(state, prm). */
  virtual void res(const Vec& state,
                   const Vec& prm,
                   Vec&       out,
                   Ctx&       ctx) const = 0;

  /** @brief Assemble dR/dstate at the supplied point. */
  virtual void assembleStateJac(const Vec& state,
                                const Vec& prm,
                                Mat&       out,
                                Ctx&       ctx) const = 0;

  /** @brief Apply (dR/dprm)^T to an adjoint vector. */
  virtual void applyParamJacT(const Vec& state,
                              const Vec& prm,
                              const Vec& adj,
                              Vec&       out,
                              Ctx&       ctx) const = 0;
};

using HostResidual   = Residual<linalg::HostCsrBackend>;
using DeviceResidual = Residual<linalg::CudaCsrBackend>;

} // namespace femx::state
