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

/** @brief Time residual decorated with constraints in one backend. */
template <class Backend>
class ConstrainedTimeResidual final : public state::TimeResidual<Backend>
{
public:
  using Base      = state::TimeResidual<Backend>;
  using Vec       = typename Base::Vec;
  using VecView   = typename Base::VecView;
  using ConstView = typename Base::ConstView;
  using Mat       = typename Base::Mat;
  using Graph     = typename Base::Graph;
  using Ctx       = typename Base::Ctx;
  using StepCtx   = typename Base::StepCtx;
  using Boundary  = BoundaryMap<Backend::space>;
  using Control   = fem::ControlMap<Backend::space>;
  using InitMap   = fem::InitialStateMap<Backend::space>;

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

  const HostCsrGraph& hostGraph() const override;
  const Graph&        graph() const override;

  const Control& controlMap() const noexcept;

  /** @brief Host-only convenience API used when rebuilding inverse metadata. */
  void setInitialStateMap(fem::HostInitialStateMap init);
  void clearInitialStateMap() noexcept;

  void initialState(ConstView prm, Vec& out, Ctx& ctx) const override;
  void addInitialStateJacobianTranspose(ConstView state_grad,
                                        VecView   out,
                                        Ctx&      ctx) const override;

  void res(const StepCtx& time, Vec& out, Ctx& ctx) const override;
  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Mat&           jac,
                    Ctx&           ctx) const override;
  void applyJacT(const StepCtx&       time,
                 state::VariableBlock wrt,
                 ConstView            adj,
                 Vec&                 out,
                 Ctx&                 ctx) const override;
  void prepareLinearSolve(const StepCtx& time,
                          Mat&           jac,
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
    ConstrainedTimeResidual<linalg::HostCsrBackend>;
using DeviceConstrainedTimeResidual =
    ConstrainedTimeResidual<linalg::CudaCsrBackend>;
#if defined(FEMX_HAS_PETSC)
using PetscConstrainedTimeResidual =
    ConstrainedTimeResidual<linalg::PetscBackend>;
#endif

} // namespace femx::assembly
