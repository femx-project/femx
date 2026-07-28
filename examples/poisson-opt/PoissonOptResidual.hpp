#pragma once

#include "PoissonOptProblem.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/fem/ElementQuadData.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/state/Residual.hpp>

namespace femx
{
namespace linalg
{
class CudaContext;
}

namespace examples::poisson_opt
{
namespace detail
{

FEMX_HOST_DEVICE inline Real controlResidual(Real state, Real ctr)
{
  return state - ctr;
}

} // namespace detail

/** @brief Assemble the controlled Poisson residual with Host data. */
class HostPoissonOptResidual final : public state::HostResidual
{
public:
  /**
   * @brief Bind to a Host Poisson optimization problem.
   *
   * @param[in] problem - Problem data kept alive while this residual is used.
   */
  explicit HostPoissonOptResidual(const PoissonOptProblem& problem);

  /** @brief Return the residual dimensions. */
  state::Dimensions dims() const override;

  /** @brief Return the Host Jacobian sparsity pattern. */
  const HostCsrPattern& hostPattern() const override;

  /**
   * @brief Assemble the controlled Poisson residual.
   *
   * @param[in] state - State vector.
   * @param[in] prm - Boundary-control parameters.
   * @param[out] out - Residual vector.
   * @param[in,out] ctx - Linear algebra context.
   */
  void assembleResidual(const HostVector<Real>& state,
                        const HostVector<Real>& prm,
                        HostVector<Real>&       out,
                        Ctx&                    ctx) const override;

  /**
   * @brief Assemble the state Jacobian.
   *
   * @param[in] state - State vector.
   * @param[in] prm - Boundary-control parameters.
   * @param[out] out - State Jacobian.
   * @param[in,out] ctx - Linear algebra context.
   */
  void assembleJacobian(const HostVector<Real>& state,
                        const HostVector<Real>& prm,
                        Jac&                    out,
                        Ctx&                    ctx) const override;

  /**
   * @brief Apply the parameter Jacobian transpose.
   *
   * @param[in] state - State vector.
   * @param[in] prm - Boundary-control parameters.
   * @param[in] adj - Residual adjoint.
   * @param[out] out - Parameter-space result.
   * @param[in,out] ctx - Linear algebra context.
   */
  void applyParamJacT(const HostVector<Real>& state,
                      const HostVector<Real>& prm,
                      const HostVector<Real>& adj,
                      HostVector<Real>&       out,
                      Ctx&                    ctx) const override;

private:
  void checkVectors(const HostVector<Real>& state,
                    const HostVector<Real>& prm) const;

  HostVector<Real> boundaryValues(
      const HostVector<Real>& prm) const;

  const PoissonOptProblem& problem_; ///< Bound Host problem.
};

/** @brief Own Device data and assemble the controlled Poisson residual on CUDA. */
class DevicePoissonOptResidual final : public state::DeviceResidual
{
public:
  /**
   * @brief Copy a Host Poisson optimization problem to Device storage.
   *
   * @param[in] problem - Source Host problem.
   * @param[in,out] ctx - CUDA context receiving the copies.
   */
  DevicePoissonOptResidual(const PoissonOptProblem& problem,
                           linalg::CudaContext&     ctx);

  /** @brief Return the residual dimensions. */
  state::Dimensions dims() const override;

  /** @brief Return the Host Jacobian sparsity pattern. */
  const HostCsrPattern& hostPattern() const override;

  /**
   * @brief Assemble the controlled Poisson residual on Device.
   *
   * @param[in] state - Device state vector.
   * @param[in] prm - Device boundary-control parameters.
   * @param[out] out - Device residual vector.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void assembleResidual(const DeviceVector<Real>& state,
                        const DeviceVector<Real>& prm,
                        DeviceVector<Real>&       out,
                        Ctx&                      ctx) const override;

  /**
   * @brief Assemble the Device state Jacobian.
   *
   * @param[in] state - Device state vector.
   * @param[in] prm - Device boundary-control parameters.
   * @param[out] out - Device state Jacobian.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void assembleJacobian(const DeviceVector<Real>& state,
                        const DeviceVector<Real>& prm,
                        Jac&                      out,
                        Ctx&                      ctx) const override;

  /**
   * @brief Apply the Device parameter Jacobian transpose.
   *
   * @param[in] state - Device state vector.
   * @param[in] prm - Device boundary-control parameters.
   * @param[in] adj - Device residual adjoint.
   * @param[out] out - Device parameter-space result.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void applyParamJacT(const DeviceVector<Real>& state,
                      const DeviceVector<Real>& prm,
                      const DeviceVector<Real>& adj,
                      DeviceVector<Real>&       out,
                      Ctx&                      ctx) const override;

private:
  void checkVectors(const DeviceVector<Real>& state,
                    const DeviceVector<Real>& prm) const;

  DeviceVectorView<const Real> boundaryValues(
      const DeviceVector<Real>& prm,
      linalg::CudaContext&      ctx) const;

  Index                       num_states_{0}; ///< Number of states.
  Index                       num_prm_{0};    ///< Number of controls.
  HostCsrPattern              h_pattern_;     ///< Canonical sparsity pattern.
  fem::DeviceMesh             mesh_;          ///< Device mesh.
  fem::DeviceElementQuadData  elem_data_;     ///< Device integration data.
  assembly::DeviceAssemblyMap assm_map_;      ///< Device assembly mapping.
  assembly::DeviceBoundaryMap boundary_map_;  ///< Device constrained rows.
  DeviceVector<Index>         ctr_dofs_;      ///< Device control rows.
  mutable DeviceVector<Real>  boundary_vals_; ///< Device boundary workspace.
};

} // namespace examples::poisson_opt
} // namespace femx
