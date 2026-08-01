#pragma once

#include "PoissonProblem.hpp"
#include <femx/linalg/SystemMatrix.hpp>
#include <femx/state/Residual.hpp>

namespace femx
{
namespace linalg
{
class CudaContext;
}

namespace examples::poisson
{

/** @brief Assemble the Poisson residual with Host data. */
class HostPoissonResidual final : public state::HostResidual
{
public:
  /**
   * @brief Bind to a Host Poisson problem.
   *
   * @param[in] problem - Problem data kept alive while this residual is used.
   */
  explicit HostPoissonResidual(const PoissonProblem& problem);

  /** @brief Return the state, parameter, and residual dimensions. */
  state::Dimensions dims() const override;

  /** @brief Return the Host Jacobian sparsity pattern. */
  const HostCsrPattern& hostPattern() const override;

  /**
   * @brief Assemble the residual vector.
   *
   * @param[in] state - State vector.
   * @param[in] prm - Parameter vector.
   * @param[out] out - Residual vector.
   * @param[in,out] ctx - Linear algebra context.
   */
  void assembleResidual(const HostVector<Real>&             state,
                        const HostVector<Real>&             prm,
                        HostVector<Real>&                   out,
                        linalg::Context<MemorySpace::Host>& ctx) const override;

  /**
   * @brief Assemble the state Jacobian.
   *
   * @param[in] state - State vector.
   * @param[in] prm - Parameter vector.
   * @param[out] out - State Jacobian.
   * @param[in,out] ctx - Linear algebra context.
   */
  void assembleJacobian(
      const HostVector<Real>&                  state,
      const HostVector<Real>&                  prm,
      linalg::SystemMatrix<MemorySpace::Host>& out,
      linalg::Context<MemorySpace::Host>&      ctx) const override;

  /**
   * @brief Apply the transpose parameter Jacobian.
   *
   * @param[in] state - State vector.
   * @param[in] prm - Parameter vector.
   * @param[in] adj - Residual adjoint.
   * @param[out] out - Parameter-space result.
   * @param[in,out] ctx - Linear algebra context.
   */
  void applyParamJacT(
      const HostVector<Real>&             state,
      const HostVector<Real>&             prm,
      const HostVector<Real>&             adj,
      HostVector<Real>&                   out,
      linalg::Context<MemorySpace::Host>& ctx) const override;

private:
  const PoissonProblem& problem_; ///< Host problem data.
};

/** @brief Own Device data and assemble the Poisson residual with CUDA. */
class CudaPoissonResidual final : public state::DeviceResidual
{
public:
  /**
   * @brief Copy a Host Poisson problem to Device storage.
   *
   * @param[in] problem - Source Host problem.
   * @param[in,out] ctx - CUDA context receiving the copies.
   */
  CudaPoissonResidual(const PoissonProblem& problem,
                      linalg::CudaContext&  ctx);

  /** @brief Return the state, parameter, and residual dimensions. */
  state::Dimensions dims() const override;

  /** @brief Return the Host Jacobian sparsity pattern. */
  const HostCsrPattern& hostPattern() const override;

  /**
   * @brief Assemble the residual vector.
   *
   * @param[in] state - Device state vector.
   * @param[in] prm - Device parameter vector.
   * @param[out] out - Device residual vector.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void assembleResidual(const DeviceVector<Real>&             state,
                        const DeviceVector<Real>&             prm,
                        DeviceVector<Real>&                   out,
                        linalg::Context<MemorySpace::Device>& ctx) const override;

  /**
   * @brief Assemble the state Jacobian.
   *
   * @param[in] state - Device state vector.
   * @param[in] prm - Device parameter vector.
   * @param[out] out - Device state Jacobian.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void assembleJacobian(
      const DeviceVector<Real>&                  state,
      const DeviceVector<Real>&                  prm,
      linalg::SystemMatrix<MemorySpace::Device>& out,
      linalg::Context<MemorySpace::Device>&      ctx) const override;

  /**
   * @brief Apply the transpose parameter Jacobian.
   *
   * @param[in] state - Device state vector.
   * @param[in] prm - Device parameter vector.
   * @param[in] adj - Device residual adjoint.
   * @param[out] out - Device parameter-space result.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void applyParamJacT(
      const DeviceVector<Real>&             state,
      const DeviceVector<Real>&             prm,
      const DeviceVector<Real>&             adj,
      DeviceVector<Real>&                   out,
      linalg::Context<MemorySpace::Device>& ctx) const override;

private:
  Index                       num_dofs_;      ///< Number of state unknowns.
  HostCsrPattern              h_pattern_;     ///< Canonical sparsity pattern.
  fem::DeviceMesh             mesh_;          ///< Device mesh data.
  fem::DeviceElementQuadData  elem_data_;     ///< Device integration data.
  assembly::DeviceAssemblyMap assm_map_;      ///< Device assembly mapping.
  assembly::DeviceBoundaryMap boundary_map_;  ///< Device constrained rows.
  DeviceVector<Real>          boundary_vals_; ///< Device boundary values.
};

} // namespace examples::poisson
} // namespace femx
