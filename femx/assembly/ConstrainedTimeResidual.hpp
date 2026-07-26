#pragma once

#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx::assembly
{

/** @brief Decorate a time residual with constraints in one memory space. */
template <MemorySpace Space>
class ConstrainedTimeResidual final : public state::TimeResidual<Space>
{
public:
  using Base      = state::TimeResidual<Space>;
  using Vec       = typename Base::Vec;
  using VecView   = typename Base::VecView;
  using ConstView = typename Base::ConstView;
  using Jac       = typename Base::Jac;
  using Ctx       = typename Base::Ctx;
  using StepCtx   = typename Base::StepCtx;
  using Boundary  = BoundaryMap<Space>;
  using Control   = fem::ControlMap<Space>;
  using InitMap   = fem::InitialStateMap<Space>;

  /**
   * @brief Decorate a non-owning Host residual.
   *
   * @param[in] base - Residual kept alive while this decorator is used.
   * @param[in] control - Host control map copied into the decorator.
   * @param[in] init - Host initial-state map copied into the decorator.
   */
  ConstrainedTimeResidual(const Base&              base,
                          fem::HostControlMap      control,
                          fem::HostInitialStateMap init = {});

  /**
   * @brief Decorate a non-owning Device residual and copy constraint data.
   *
   * @param[in] base - Residual kept alive while this decorator is used.
   * @param[in] control - Host control map copied to Device storage.
   * @param[in] init - Host initial-state map copied to Device storage.
   * @param[in,out] ctx - Device context receiving the copies.
   */
  ConstrainedTimeResidual(const Base&              base,
                          fem::HostControlMap      control,
                          fem::HostInitialStateMap init,
                          Ctx&                     ctx);

  state::TimeDims dims() const override;

  const HostCsrPattern& hostPattern() const override;

  const Control& controlMap() const noexcept;

  /** @brief Host-only convenience API used when rebuilding inverse metadata. */
  void setInitialStateMap(fem::HostInitialStateMap init);
  void clearInitialStateMap() noexcept;

  void initialState(ConstView prm, Vec& out, Ctx& ctx) const override;
  void addInitialStateJacT(ConstView state_grad,
                           VecView   out,
                           Ctx&      ctx) const override;

  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Jac&           jac,
                    Ctx&           ctx) const override;
  void applyJacT(const StepCtx&       time,
                 state::VariableBlock wrt,
                 ConstView            adj,
                 Vec&                 out,
                 Ctx&                 ctx) const override;

  /** @copydoc state::TimeResidual<Space>::setup */
  void setup(const StepCtx& time,
             Jac&           jac,
             Vec&           rhs,
             Ctx&           ctx) const override;

private:
  StepCtx baseCtx(const StepCtx& time) const;

  void initDims(const fem::HostControlMap&      control,
                const fem::HostInitialStateMap& init);
  void checkCtx(const StepCtx& time) const;
  void checkInitMap(const fem::HostInitialStateMap& map) const;

  const Base&     base_;          ///< Decorated residual.
  Control         control_;       ///< Control mapping in the target memory space.
  InitMap         init_;          ///< Initial-state mapping in the target memory space.
  Boundary        boundary_;      ///< Constrained degrees of freedom.
  Vec             base_prm_;      ///< Parameters passed to the decorated residual.
  mutable Vec     base_adj_;      ///< Adjoint for the decorated residual.
  mutable Vec     boundary_vals_; ///< Evaluated boundary values.
  state::TimeDims base_dims_;     ///< Dimensions of the decorated residual.
  state::TimeDims dims_;          ///< Dimensions including control parameters.
};

using HostConstrainedTimeResidual   = ConstrainedTimeResidual<MemorySpace::Host>;
using DeviceConstrainedTimeResidual = ConstrainedTimeResidual<MemorySpace::Device>;
} // namespace femx::assembly
