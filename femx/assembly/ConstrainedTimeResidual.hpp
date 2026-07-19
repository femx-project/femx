#pragma once

#include <memory>

#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/state/TimeResidual.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScBackend.hpp>
#endif

namespace femx::assembly
{

/** @brief Host-state residual decorated with time-dependent constraints. */
template <class Backend>
class ConstrainedTimeResidual final : public state::TimeResidual<Backend>
{
  static_assert(Backend::space == MemorySpace::Host,
                "Generic ConstrainedTimeResidual requires Host state storage");

public:
  using Base      = state::TimeResidual<Backend>;
  using Mat       = typename Base::Mat;
  using Ctx       = typename Base::Ctx;
  using ConstView = typename Base::ConstView;

  ConstrainedTimeResidual(const Base&              base,
                          fem::HostControlMap      control,
                          fem::HostInitialStateMap init_state = {});

  state::TimeDims dims() const override;

  const HostCsrGraph&         hostGraph() const override;
  const typename Base::Graph& graph() const override;

  const fem::HostControlMap& controlMap() const noexcept;
  void                       setInitialStateMap(fem::HostInitialStateMap init_state);
  void                       clearInitialStateMap() noexcept;

  void initialState(ConstView prm, HostVector& out, Ctx& ctx) const override;
  void addInitialStateJacobianTranspose(ConstView      state_grad,
                                        HostVectorView out,
                                        Ctx&           ctx) const override;

  void res(const state::HostTimeContext& time,
           HostVector&                   out,
           Ctx&                          ctx) const override;
  void assemble(const state::HostTimeContext& time,
                state::VariableBlock          wrt,
                HostVector&                   res,
                Mat&                          jac,
                Ctx&                          ctx) const override;
  void applyJac(const state::HostTimeContext& time,
                state::VariableBlock          wrt,
                ConstView                     dir,
                HostVector&                   out,
                Ctx&                          ctx) const override;
  void applyJacT(const state::HostTimeContext& time,
                 state::VariableBlock          wrt,
                 ConstView                     adj,
                 HostVector&                   out,
                 Ctx&                          ctx) const override;
  void assembleJac(const state::HostTimeContext& time,
                   state::VariableBlock          wrt,
                   Mat&                          out,
                   Ctx&                          ctx) const override;
  void prepareLinearSolve(const state::HostTimeContext& time,
                          state::VariableBlock          wrt,
                          Mat&                          jac,
                          HostVector&                   rhs,
                          Ctx&                          ctx) const override;

private:
  state::HostTimeContext baseContext(
      const state::HostTimeContext& time) const;
  void checkContext(const state::HostTimeContext& time) const;
  void checkInitialStateMap(const fem::HostInitialStateMap& map) const;

  const Base&              base_;
  fem::HostControlMap      control_;
  fem::HostInitialStateMap init_state_;
  HostBoundaryMap          boundary_;
  HostVector               base_prm_;
  mutable HostVector       boundary_vals_;
  state::TimeDims          base_dims_;
  state::TimeDims          dims_;
};

/** @brief CUDA residual decorated with time-dependent constraints. */
template <>
class ConstrainedTimeResidual<linalg::CudaCsrBackend> final
  : public state::DeviceTimeResidual
{
public:
  using Base = state::DeviceTimeResidual;

  ConstrainedTimeResidual(std::unique_ptr<Base>    base,
                          fem::HostControlMap      control,
                          fem::HostInitialStateMap init_state,
                          CudaContext&             ctx);
  ~ConstrainedTimeResidual() override;

  state::TimeDims       dims() const override;
  const HostCsrGraph&   hostGraph() const override;
  const DeviceCsrGraph& graph() const override;

  void initialState(DeviceConstVectorView prm,
                    DeviceVector&         out,
                    CudaContext&          ctx) const override;
  void addInitialStateJacobianTranspose(DeviceConstVectorView state_grad,
                                        DeviceVectorView      out,
                                        CudaContext&          ctx) const override;
  void res(const state::DeviceTimeContext& time,
           DeviceVector&                   out,
           CudaContext&                    ctx) const override;
  void assemble(const state::DeviceTimeContext& time,
                state::VariableBlock            wrt,
                DeviceVector&                   res,
                DeviceCsrMatrix&                jac,
                CudaContext&                    ctx) const override;
  void applyJac(const state::DeviceTimeContext& time,
                state::VariableBlock            wrt,
                DeviceConstVectorView           dir,
                DeviceVector&                   out,
                CudaContext&                    ctx) const override;
  void applyJacT(const state::DeviceTimeContext& time,
                 state::VariableBlock            wrt,
                 DeviceConstVectorView           adj,
                 DeviceVector&                   out,
                 CudaContext&                    ctx) const override;
  void assembleJac(const state::DeviceTimeContext& time,
                   state::VariableBlock            wrt,
                   DeviceCsrMatrix&                out,
                   CudaContext&                    ctx) const override;
  void prepareLinearSolve(const state::DeviceTimeContext& time,
                          state::VariableBlock            wrt,
                          DeviceCsrMatrix&                jac,
                          DeviceVector&                   rhs,
                          CudaContext&                    ctx) const override;

private:
  state::DeviceTimeContext baseContext(
      const state::DeviceTimeContext& time) const;
  void checkContext(const state::DeviceTimeContext& time) const;
  void checkParameters(DeviceConstVectorView prm) const;

  std::unique_ptr<Base>      base_;
  state::TimeDims            base_dims_;
  state::TimeDims            dims_;
  DeviceBoundaryMap          boundary_;
  fem::DeviceControlMap      control_;
  fem::DeviceInitialStateMap init_state_;
  mutable DeviceVector       boundary_vals_;
  bool                       has_init_state_{false};
};

using HostConstrainedTimeResidual =
    ConstrainedTimeResidual<linalg::HostCsrBackend>;
using DeviceConstrainedTimeResidual =
    ConstrainedTimeResidual<linalg::CudaCsrBackend>;
#if defined(FEMX_HAS_PETSC)
using PetscConstrainedTimeResidual =
    ConstrainedTimeResidual<linalg::PetscBackend>;
#endif

} // namespace femx::assembly
