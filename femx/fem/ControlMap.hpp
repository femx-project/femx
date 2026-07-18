#pragma once

#include <femx/common/Context.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

/** @brief Flat fixed and controlled Dirichlet data in one memory space. */
template <MemorySpace Space>
struct ControlMapView
{
  Index num_steps{0};   ///< Number of residual time steps.
  Index num_states{0};  ///< Full state and residual size.
  Index num_prm{0};     ///< Full parameter-vector size.
  Index num_ctr{0};     ///< Number of controlled boundary DOFs.
  Index num_fixed{0};   ///< Number of fixed boundary DOFs.
  Index num_ctr_prm{0}; ///< Control parameters on one time level.
  Index ctr_off{0};     ///< First control entry in the full parameters.

  const Index* dofs{nullptr};        ///< Control then fixed state DOFs.
  const Index* ctr_offsets{nullptr}; ///< Row offsets of the control map.
  const Index* ctr_cols{nullptr};    ///< Control-map columns.
  const Real*  ctr_wts{nullptr};     ///< Control-map values.
  const Real*  fixed_vals{nullptr};  ///< Step-major fixed values.
  const Index* lower{nullptr};       ///< Lower control level per step.
  const Index* upper{nullptr};       ///< Upper control level per step.
  const Real*  upper_wts{nullptr};   ///< Upper interpolation weights.

  /** @brief Total number of constrained state DOFs. */
  FEMX_HOST_DEVICE Index numBcs() const
  {
    return num_ctr + num_fixed;
  }

  /** @brief Constrained state DOF at boundary-map index `ib`. */
  FEMX_HOST_DEVICE Index dof(Index ib) const
  {
    return dofs[ib];
  }

  /** @brief First control-map entry for controlled boundary row `i`. */
  FEMX_HOST_DEVICE Index ctrBegin(Index i) const
  {
    return ctr_offsets[i];
  }

  /** @brief One-past-last control-map entry for boundary row `i`. */
  FEMX_HOST_DEVICE Index ctrEnd(Index i) const
  {
    return ctr_offsets[i + 1];
  }

  /** @brief Parameter column used by flattened control entry `k`. */
  FEMX_HOST_DEVICE Index ctrCol(Index k) const
  {
    return ctr_cols[k];
  }

  /** @brief Weight used by flattened control entry `k`. */
  FEMX_HOST_DEVICE Real ctrWt(Index k) const
  {
    return ctr_wts[k];
  }

  /** @brief Full parameter index for control `i` on time level `level`. */
  FEMX_HOST_DEVICE Index ctrIndex(Index level, Index i) const
  {
    return ctr_off + level * num_ctr_prm + i;
  }

  /** @brief Lower control level at residual step `step`. */
  FEMX_HOST_DEVICE Index lowerLevel(Index step) const
  {
    return lower[step];
  }

  /** @brief Upper control level at residual step `step`. */
  FEMX_HOST_DEVICE Index upperLevel(Index step) const
  {
    return upper[step];
  }

  /** @brief Upper-level interpolation weight at residual step `step`. */
  FEMX_HOST_DEVICE Real upperWt(Index step) const
  {
    return upper_wts[step];
  }

  /** @brief Fixed value `i` at residual step `step`. */
  FEMX_HOST_DEVICE Real fixedValue(Index step, Index i) const
  {
    return fixed_vals[step * num_fixed + i];
  }
};

/** @brief Owner of persistent fixed and controlled Dirichlet data. */
template <MemorySpace Space>
class ControlMap
{
public:
  ControlMap() = default;

  ControlMap(const ControlMap&)                = default;
  ControlMap(ControlMap&&) noexcept            = default;
  ControlMap& operator=(const ControlMap&)     = default;
  ControlMap& operator=(ControlMap&&) noexcept = default;

  /** @brief Number of residual time steps. */
  Index numSteps() const noexcept
  {
    return num_steps_;
  }

  /** @brief Full state and residual size. */
  Index numStates() const noexcept
  {
    return num_states_;
  }

  /** @brief Full parameter-vector size. */
  Index numParams() const noexcept
  {
    return num_prm_;
  }

  /** @brief Number of control parameters on one time level. */
  Index numControlParams() const noexcept
  {
    return num_ctr_prm_;
  }

  /** @brief Number of constrained DOFs in control-then-fixed order. */
  Index numBcs() const noexcept
  {
    return dofs_.size();
  }

  /**
   * @brief Constrained DOFs in the order used by controlVals().
   *
   * A Host map's result can be passed directly to makeBoundaryMap().
   */
  const Vector<Space, Index>& dofs() const noexcept
  {
    return dofs_;
  }

  /** @brief Return a non-owning kernel view valid while this map is alive. */
  ControlMapView<Space> view() const noexcept
  {
    return {num_steps_,
            num_states_,
            num_prm_,
            num_ctr_,
            num_fixed_,
            num_ctr_prm_,
            ctr_off_,
            dofs_.data(),
            ctr_offsets_.data(),
            ctr_cols_.data(),
            ctr_wts_.data(),
            fixed_vals_.data(),
            lower_.data(),
            upper_.data(),
            upper_wts_.data()};
  }

private:
  friend ControlMap<MemorySpace::Host> makeControlMap(
      Index,
      Index,
      const DirichletControl&,
      Array<Index>,
      HostVector,
      Array<LinearInterpolation>,
      Index,
      Index);

  friend void copy(const ControlMap<MemorySpace::Host>&,
                   ControlMap<MemorySpace::Device>&,
                   CudaContext&);

  Index num_steps_{0};
  Index num_states_{0};
  Index num_prm_{0};
  Index num_ctr_{0};
  Index num_fixed_{0};
  Index num_ctr_prm_{0};
  Index ctr_off_{0};

  Vector<Space, Index> dofs_;
  Vector<Space, Index> ctr_offsets_;
  Vector<Space, Index> ctr_cols_;
  Vector<Space, Real>  ctr_wts_;
  Vector<Space, Real>  fixed_vals_;
  Vector<Space, Index> lower_;
  Vector<Space, Index> upper_;
  Vector<Space, Real>  upper_wts_;
};

using HostControlMap   = ControlMap<MemorySpace::Host>;
using DeviceControlMap = ControlMap<MemorySpace::Device>;

/**
 * @brief Build reusable Dirichlet data on Host.
 *
 * Boundary order is always controlled DOFs followed by fixed DOFs. An empty
 * `time` array uses one control level per residual step. Fixed values may be
 * empty, one value per fixed DOF, or step-major values for all steps.
 * @param num_steps Number of residual steps.
 * @param num_states Full state and residual size.
 * @param ctr Sparse boundary-control topology.
 * @param fixed_dofs Fixed constrained DOFs.
 * @param fixed_vals Constant or step-major fixed values.
 * @param time Control interpolation stencil for each residual step.
 * @param ctr_off First time-control entry in the full parameter vector.
 * @param num_prm Full parameter size, or -1 to infer its minimum size.
 */
HostControlMap makeControlMap(
    Index                      num_steps,
    Index                      num_states,
    const DirichletControl&    ctr,
    Array<Index>               fixed_dofs,
    HostVector                 fixed_vals,
    Array<LinearInterpolation> time,
    Index                      ctr_off,
    Index                      num_prm = -1);

/** @brief Explicitly copy persistent Host control data to Device storage. */
void copy(const HostControlMap& src,
          DeviceControlMap&     dst,
          CudaContext&          ctx);

/**
 * @brief Evaluate controlled and fixed values into preallocated Host storage.
 *
 * Every output entry is overwritten in control-then-fixed boundary order.
 */
void controlVals(ControlMapView<MemorySpace::Host> map,
                 Index                             step,
                 HostConstVectorView               prm,
                 HostVectorView                    out);

/** @brief Asynchronous Device equivalent of controlVals(). */
void controlVals(ControlMapView<MemorySpace::Device> map,
                 Index                               step,
                 DeviceConstVectorView               prm,
                 DeviceVectorView                    out,
                 CudaContext&                        ctx);

/**
 * @brief Apply the boundary residual parameter Jacobian on Host.
 *
 * `out` is overwritten with zero except for controlled rows, which receive
 * `-P` times the time-interpolated parameter direction.
 */
void controlJac(ControlMapView<MemorySpace::Host> map,
                Index                             step,
                HostConstVectorView               dir,
                HostVectorView                    out);

/** @brief Asynchronous Device equivalent of controlJac(). */
void controlJac(ControlMapView<MemorySpace::Device> map,
                Index                               step,
                DeviceConstVectorView               dir,
                DeviceVectorView                    out,
                CudaContext&                        ctx);

/**
 * @brief Add the Host product `J_prm^T adj` to a parameter gradient.
 *
 * The residual Jacobian includes the minus sign from `state - P * control`.
 */
void addControlJacT(ControlMapView<MemorySpace::Host> map,
                    Index                             step,
                    HostConstVectorView               adj,
                    HostVectorView                    grad);

/** @brief Asynchronous Device equivalent of addControlJacT(). */
void addControlJacT(ControlMapView<MemorySpace::Device> map,
                    Index                               step,
                    DeviceConstVectorView               adj,
                    DeviceVectorView                    grad,
                    CudaContext&                        ctx);

/** @brief Flat affine initial-state parameterization in one memory space. */
template <MemorySpace Space>
struct InitialStateMapView
{
  Index num_states{0}; ///< Full state size.
  Index num_prm{0};    ///< Full parameter-vector size.
  Index num_modes{0};  ///< Number of affine modes.
  Index num_ctr{0};    ///< Number of controlled initial-state DOFs.
  Index init_off{0};   ///< First modal parameter.
  Index ctr_off{0};    ///< First level-zero control parameter.

  const Real*  mean{nullptr};        ///< Mean initial state.
  const Real*  modes{nullptr};       ///< Row-major state-by-mode matrix.
  const Index* ctr_dofs{nullptr};    ///< Controlled global state DOFs.
  const Index* ctr_offsets{nullptr}; ///< Row offsets of the control map.
  const Index* ctr_cols{nullptr};    ///< Control-map columns.
  const Real*  ctr_wts{nullptr};     ///< Control-map values.

  /** @brief First control-map entry for controlled state row `i`. */
  FEMX_HOST_DEVICE Index ctrBegin(Index i) const
  {
    return ctr_offsets[i];
  }

  /** @brief One-past-last control-map entry for controlled row `i`. */
  FEMX_HOST_DEVICE Index ctrEnd(Index i) const
  {
    return ctr_offsets[i + 1];
  }

  /** @brief Row-major modal entry `(row, col)`. */
  FEMX_HOST_DEVICE Real mode(Index row, Index col) const
  {
    return modes[row * num_modes + col];
  }
};

/** @brief Owner of a persistent affine initial-state parameterization. */
template <MemorySpace Space>
class InitialStateMap
{
public:
  InitialStateMap() = default;

  InitialStateMap(const InitialStateMap&)                = default;
  InitialStateMap(InitialStateMap&&) noexcept            = default;
  InitialStateMap& operator=(const InitialStateMap&)     = default;
  InitialStateMap& operator=(InitialStateMap&&) noexcept = default;

  /** @brief Full state size. */
  Index numStates() const noexcept
  {
    return num_states_;
  }

  /** @brief Full parameter-vector size. */
  Index numParams() const noexcept
  {
    return num_prm_;
  }

  /** @brief Number of modal initial-state parameters. */
  Index numModes() const noexcept
  {
    return num_modes_;
  }

  /** @brief Return a non-owning kernel view valid while this map is alive. */
  InitialStateMapView<Space> view() const noexcept
  {
    return {num_states_,
            num_prm_,
            num_modes_,
            num_ctr_,
            init_off_,
            ctr_off_,
            mean_.data(),
            modes_.data(),
            ctr_dofs_.data(),
            ctr_offsets_.data(),
            ctr_cols_.data(),
            ctr_wts_.data()};
  }

private:
  friend InitialStateMap<MemorySpace::Host> makeInitialStateMap(
      HostVector,
      DenseMatrix,
      const DirichletControl&,
      Index,
      Index,
      Index);

  friend void copy(const InitialStateMap<MemorySpace::Host>&,
                   InitialStateMap<MemorySpace::Device>&,
                   CudaContext&);

  Index num_states_{0};
  Index num_prm_{0};
  Index num_modes_{0};
  Index num_ctr_{0};
  Index init_off_{0};
  Index ctr_off_{0};

  Vector<Space, Real>  mean_;
  Vector<Space, Real>  modes_;
  Vector<Space, Index> ctr_dofs_;
  Vector<Space, Index> ctr_offsets_;
  Vector<Space, Index> ctr_cols_;
  Vector<Space, Real>  ctr_wts_;
};

using HostInitialStateMap   = InitialStateMap<MemorySpace::Host>;
using DeviceInitialStateMap = InitialStateMap<MemorySpace::Device>;

/**
 * @brief Build an affine initial-state map on Host.
 *
 * The mean plus modal state is formed first. Controlled DOFs are then
 * overwritten by `P` times the level-zero control block at `ctr_off`.
 * @param mean Mean initial state.
 * @param modes Row-major state modes supplied as a dense matrix.
 * @param ctr Boundary-control topology applied at the initial level.
 * @param init_off First modal entry in the full parameter vector.
 * @param ctr_off First level-zero control entry in the parameter vector.
 * @param num_prm Full parameter-vector size.
 */
HostInitialStateMap makeInitialStateMap(HostVector              mean,
                                        DenseMatrix             modes,
                                        const DirichletControl& ctr,
                                        Index                   init_off,
                                        Index                   ctr_off,
                                        Index                   num_prm);

/** @brief Explicitly copy a Host initial-state map to Device storage. */
void copy(const HostInitialStateMap& src,
          DeviceInitialStateMap&     dst,
          CudaContext&               ctx);

/** @brief Evaluate an affine initial state into preallocated Host storage. */
void initialState(InitialStateMapView<MemorySpace::Host> map,
                  HostConstVectorView                    prm,
                  HostVectorView                         out);

/** @brief Asynchronous Device equivalent of initialState(). */
void initialState(InitialStateMapView<MemorySpace::Device> map,
                  DeviceConstVectorView                    prm,
                  DeviceVectorView                         out,
                  CudaContext&                             ctx);

/** @brief Add the Host initial-state transpose product to `grad`. */
void addInitialJacT(InitialStateMapView<MemorySpace::Host> map,
                    HostConstVectorView                    adj,
                    HostVectorView                         grad);

/** @brief Asynchronously add the Device initial-state transpose product. */
void addInitialJacT(InitialStateMapView<MemorySpace::Device> map,
                    DeviceConstVectorView                    adj,
                    DeviceVectorView                         grad,
                    CudaContext&                             ctx);

} // namespace fem
} // namespace femx
