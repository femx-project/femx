#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/SystemMatrix.hpp>

namespace femx::state
{

/**
 * @brief Sizes for a parameter-dependent stationary residual.
 */
struct Dimensions
{
  Index num_states{0};
  Index num_param{0};
  Index num_res{0};
};

/**
 * @brief Define a stationary residual in one memory space.
 */
template <MemorySpace Space>
class Residual
{
public:
  static constexpr MemorySpace space = Space;

  using Vec = Vector<Space, Real>;
  using Jac = linalg::SystemMatrix<Space>;
  using Ctx = linalg::Context<space>;

  virtual ~Residual() = default;

  virtual Dimensions dims() const = 0;

  /**
   * @brief Return the canonical Host Jacobian pattern.
   */
  virtual const HostCsrPattern& hostPattern() const = 0;

  /**
   * @brief Assemble the residual at a state and parameter point.
   *
   * @param[in]     state - State vector.
   * @param[in]     prm   - Parameter vector.
   * @param[out]    out   - Assembled residual.
   * @param[in,out] ctx   - Linear algebra context.
   */
  virtual void assembleResidual(const Vec& state,
                                const Vec& prm,
                                Vec&       out,
                                Ctx&       ctx) const = 0;

  /**
   * @brief Assemble the state Jacobian at a state and parameter point.
   *
   * @param[in]     state - State vector.
   * @param[in]     prm   - Parameter vector.
   * @param[in,out] out   - State Jacobian receiving assembled entries.
   * @param[in,out] ctx   - Linear algebra context.
   */
  virtual void assembleJacobian(const Vec& state,
                                const Vec& prm,
                                Jac&       out,
                                Ctx&       ctx) const = 0;

  /**
   * @brief Apply (dR/dprm)^T to an adjoint vector.
   */
  virtual void applyParamJacT(const Vec& state,
                              const Vec& prm,
                              const Vec& adj,
                              Vec&       out,
                              Ctx&       ctx) const = 0;
};

using HostResidual   = Residual<MemorySpace::Host>;
using DeviceResidual = Residual<MemorySpace::Device>;

} // namespace femx::state
