#pragma once

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/fem/ElementQuadData.hpp>
#include <femx/model/navier/NavierElementKernel.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
namespace linalg
{
class CudaContext;
}

namespace model::navier
{

class NavierModel;

/** @brief Assemble the Navier-Stokes time residual with Host data. */
class HostNavierResidual final : public state::HostTimeResidual
{
public:
  /**
   * @brief Bind to a Host Navier-Stokes model.
   *
   * @param[in] model - Model data kept alive while this residual is used.
   */
  explicit HostNavierResidual(const NavierModel& model);

  /** @brief Return the time-residual dimensions. */
  state::TimeDims dims() const override;

  /** @brief Return the Host Jacobian sparsity pattern. */
  const HostCsrPattern& hostPattern() const override;

  /**
   * @brief Return the parameter-free zero initial state.
   *
   * @param[in] prm - Empty parameter vector.
   * @param[out] out - Initial state.
   * @param[in,out] ctx - Linear algebra context.
   */
  void initialState(ConstView prm,
                    Vec&      out,
                    Ctx&      ctx) const override;

  /**
   * @brief Assemble the next-state residual and Jacobian.
   *
   * @param[in] time - Time-step context.
   * @param[out] res - Next-state residual.
   * @param[out] jac - Next-state Jacobian.
   * @param[in,out] ctx - Linear algebra context.
   */
  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Jac&           jac,
                    Ctx&           ctx) const override;

  /**
   * @brief Apply a history or parameter Jacobian transpose.
   *
   * @param[in] time - Time-step context.
   * @param[in] with_respect_to - Differentiated variable block.
   * @param[in] adj - Residual adjoint.
   * @param[out] out - Transpose-product result.
   * @param[in,out] ctx - Linear algebra context.
   */
  void applyJacT(const StepCtx&       time,
                 state::VariableBlock with_respect_to,
                 ConstView            adj,
                 Vec&                 out,
                 Ctx&                 ctx) const override;

private:
  const NavierModel& model_; ///< Host model data.
};

/** @brief Own Device data and assemble the Navier-Stokes residual with CUDA. */
class DeviceNavierResidual final : public state::DeviceTimeResidual
{
public:
  /**
   * @brief Copy a Host Navier-Stokes model to Device storage.
   *
   * @param[in] model - Source Host model.
   * @param[in,out] ctx - CUDA context receiving the copies.
   */
  DeviceNavierResidual(const NavierModel&   model,
                       linalg::CudaContext& ctx);

  /** @brief Return the time-residual dimensions. */
  state::TimeDims dims() const override;

  /** @brief Return the Host Jacobian sparsity pattern. */
  const HostCsrPattern& hostPattern() const override;

  /**
   * @brief Return the parameter-free zero initial state.
   *
   * @param[in] prm - Empty Device parameter vector.
   * @param[out] out - Device initial state.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void initialState(ConstView prm,
                    Vec&      out,
                    Ctx&      ctx) const override;

  /**
   * @brief Assemble the next-state residual and Jacobian.
   *
   * @param[in] time - Device time-step context.
   * @param[out] res - Device next-state residual.
   * @param[out] jac - Device next-state Jacobian.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Jac&           jac,
                    Ctx&           ctx) const override;

  /**
   * @brief Apply a history or parameter Jacobian transpose.
   *
   * @param[in] time - Device time-step context.
   * @param[in] with_respect_to - Differentiated variable block.
   * @param[in] adj - Device residual adjoint.
   * @param[out] out - Device transpose-product result.
   * @param[in,out] ctx - Device linear algebra context.
   */
  void applyJacT(const StepCtx&       time,
                 state::VariableBlock with_respect_to,
                 ConstView            adj,
                 Vec&                 out,
                 Ctx&                 ctx) const override;

private:
  Index                       num_steps_; ///< Number of time steps.
  HostCsrPattern              h_pattern_; ///< Canonical sparsity pattern.
  fem::DeviceElementQuadData  elem_data_; ///< Device integration data.
  assembly::DeviceAssemblyMap assm_map_;  ///< Device assembly mapping.
  DeviceNavierElementKernel   kernel_;    ///< Device element operator.
};

} // namespace model::navier
} // namespace femx
