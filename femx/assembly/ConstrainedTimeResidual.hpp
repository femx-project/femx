#pragma once

#include <memory>

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

  /** @brief Decorate a non-owning Host residual. */
  ConstrainedTimeResidual(const Base&              base,
                          fem::HostControlMap      control,
                          fem::HostInitialStateMap init = {});

  /** @brief Copy constraint data and take ownership of a Device residual. */
  ConstrainedTimeResidual(std::unique_ptr<Base>    base,
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
  void prepareLinearSolve(const StepCtx& time,
                          Jac&           jac,
                          Vec&           rhs,
                          Ctx&           ctx) const override;

private:
  StepCtx baseCtx(const StepCtx& time) const;

  void initDims(const fem::HostControlMap&      control,
                const fem::HostInitialStateMap& init);
  void checkCtx(const StepCtx& time) const;
  void checkInitMap(const fem::HostInitialStateMap& map) const;

  std::unique_ptr<Base> owned_base_;
  const Base*           base_{nullptr};
  Control               control_;
  InitMap               init_;
  Boundary              boundary_;
  Vec                   base_prm_;
  mutable Vec           base_adj_;
  mutable Vec           boundary_vals_;
  state::TimeDims       base_dims_;
  state::TimeDims       dims_;
};

using HostConstrainedTimeResidual =
    ConstrainedTimeResidual<MemorySpace::Host>;
using DeviceConstrainedTimeResidual =
    ConstrainedTimeResidual<MemorySpace::Device>;
} // namespace femx::assembly
