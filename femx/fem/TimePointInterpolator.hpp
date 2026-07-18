#pragma once

#include <femx/common/Context.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

using inverse::DeviceTimeObservationOperator;
using inverse::TimeObservationOperator;

/** @brief Non-owning flat interpolation stencils in one memory space. */
template <MemorySpace Space>
class PointInterpolatorView
{
public:
  FEMX_HOST_DEVICE PointInterpolatorView() = default;

  FEMX_HOST_DEVICE PointInterpolatorView(
      Index                          num_states,
      VectorView<Space, const Index> offsets,
      VectorView<Space, const Index> dofs,
      VectorView<Space, const Real>  wts)
    : num_states_(num_states),
      offsets_(offsets),
      dofs_(dofs),
      wts_(wts)
  {
  }

  /** @brief Number of entries in an input state vector. */
  FEMX_HOST_DEVICE Index numStates() const
  {
    return num_states_;
  }

  /** @brief Number of point-component observations. */
  FEMX_HOST_DEVICE Index numObservations() const
  {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
  }

  /** @brief Number of flattened stencil entries. */
  FEMX_HOST_DEVICE Index numEntries() const
  {
    return dofs_.size();
  }

  /** @brief First flat entry for observation `i`. */
  FEMX_HOST_DEVICE Index begin(Index i) const
  {
    return offsets_[i];
  }

  /** @brief One-past-last flat entry for observation `i`. */
  FEMX_HOST_DEVICE Index end(Index i) const
  {
    return offsets_[i + 1];
  }

  /** @brief State DOF used by flat entry `k`. */
  FEMX_HOST_DEVICE Index dof(Index k) const
  {
    return dofs_[k];
  }

  /** @brief Interpolation weight used by flat entry `k`. */
  FEMX_HOST_DEVICE Real wt(Index k) const
  {
    return wts_[k];
  }

  /** @brief Evaluate one stencil from an address in the same memory space. */
  FEMX_HOST_DEVICE Real eval(Index i, const Real* state) const
  {
    Real val = 0.0;
    for (Index k = begin(i); k < end(i); ++k)
    {
      val += wt(k) * state[dof(k)];
    }
    return val;
  }

private:
  Index                          num_states_{0};
  VectorView<Space, const Index> offsets_;
  VectorView<Space, const Index> dofs_;
  VectorView<Space, const Real>  wts_;
};

template <MemorySpace Space>
class PointInterpolatorData;

class TimePointInterpolator;
class DeviceTimePointInterpolator;

using HostPointInterpolatorData =
    PointInterpolatorData<MemorySpace::Host>;
using DevicePointInterpolatorData =
    PointInterpolatorData<MemorySpace::Device>;

/** @brief Memory-space owner of reusable flat interpolation stencils. */
template <MemorySpace Space>
class PointInterpolatorData
{
public:
  PointInterpolatorData() = default;

  PointInterpolatorData(const PointInterpolatorData&)                = default;
  PointInterpolatorData(PointInterpolatorData&&) noexcept            = default;
  PointInterpolatorData& operator=(const PointInterpolatorData&)     = default;
  PointInterpolatorData& operator=(PointInterpolatorData&&) noexcept = default;

  /** @brief Number of entries in one state vector. */
  Index numStates() const noexcept
  {
    return num_states_;
  }

  /** @brief Number of point-component observations. */
  Index numObservations() const noexcept
  {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
  }

  /** @brief Number of flattened interpolation entries. */
  Index numEntries() const noexcept
  {
    return dofs_.size();
  }

  /** @brief Return a kernel view valid while this owner is alive. */
  PointInterpolatorView<Space> view() const noexcept
  {
    return {num_states_, offsets_.view(), dofs_.view(), wts_.view()};
  }

private:
  friend class TimePointInterpolator;

  friend void copy(const HostPointInterpolatorData& src,
                   DevicePointInterpolatorData&     dst,
                   CudaContext&                     ctx);

  Index                num_states_{0};
  Vector<Space, Index> offsets_;
  Vector<Space, Index> dofs_;
  Vector<Space, Real>  wts_;
};

/**
 * @brief Explicitly copy reusable Host stencils to Device storage.
 *
 * The source must remain alive until work queued on `ctx` has consumed the
 * copy or the context has been synchronized.
 */
inline void copy(const HostPointInterpolatorData& src,
                 DevicePointInterpolatorData&     dst,
                 CudaContext&                     ctx)
{
  dst.num_states_ = src.num_states_;
  femx::copy(src.offsets_, dst.offsets_, ctx);
  femx::copy(src.dofs_, dst.dofs_, ctx);
  femx::copy(src.wts_, dst.wts_, ctx);
}

/**
 * @brief Evaluate all Host observation stencils into preallocated `out`.
 *
 * Every output entry is overwritten. No storage is allocated or resized.
 */
void observe(PointInterpolatorView<MemorySpace::Host> data,
             HostConstVectorView                      state,
             HostVectorView                           out);

/**
 * @brief Add the Host state-transpose product to preallocated `out`.
 *
 * This operation accumulates `H^T dir`; callers choose when to zero `out`.
 */
void addStateJacT(PointInterpolatorView<MemorySpace::Host> data,
                  HostConstVectorView                      dir,
                  HostVectorView                           out);

/**
 * @brief Asynchronously evaluate Device stencils into preallocated `out`.
 *
 * Every output entry is overwritten on `ctx`. No storage is allocated,
 * resized, or mirrored.
 */
void observe(PointInterpolatorView<MemorySpace::Device> data,
             DeviceConstVectorView                      state,
             DeviceVectorView                           out,
             CudaContext&                               ctx);

/**
 * @brief Asynchronously add the Device state-transpose product to `out`.
 *
 * `out` must already have `numStates()` entries. The kernel accumulates into
 * its current values, so callers explicitly control zeroing and reuse.
 */
void addStateJacT(PointInterpolatorView<MemorySpace::Device> data,
                  DeviceConstVectorView                      dir,
                  DeviceVectorView                           out,
                  CudaContext&                               ctx);

/**
 * @brief Explicitly initialized Device point observation operator.
 *
 * Construct an empty owner, then call copy(TimePointInterpolator, ...) before
 * use. Its operations consume preallocated Device views without mirroring.
 */
class DeviceTimePointInterpolator final
  : public DeviceTimeObservationOperator
{
public:
  DeviceTimePointInterpolator() = default;

  DeviceTimePointInterpolator(const DeviceTimePointInterpolator&) = delete;
  DeviceTimePointInterpolator& operator=(
      const DeviceTimePointInterpolator&) = delete;
  DeviceTimePointInterpolator(DeviceTimePointInterpolator&&) noexcept =
      default;
  DeviceTimePointInterpolator& operator=(
      DeviceTimePointInterpolator&&) noexcept = default;

  /** @brief Number of residual time steps. */
  Index numSteps() const override;

  /** @brief Number of entries in one state vector. */
  Index numStates() const override;

  /** @brief Number of point-component observations. */
  Index numObservations() const override;

  void observe(Index                 level,
               DeviceConstVectorView state,
               DeviceVectorView      out,
               CudaContext&          ctx) const override;

  void addStateJacT(Index                 level,
                    DeviceConstVectorView dir,
                    DeviceVectorView      out,
                    CudaContext&          ctx) const override;

private:
  friend void copy(const TimePointInterpolator& src,
                   DeviceTimePointInterpolator& dst,
                   CudaContext&                 ctx);

  void checkLevel(Index level) const;

  Index                       num_steps_{-1};
  DevicePointInterpolatorData data_;
};

/**
 * @brief Interpolates field components at physical points on each time level.
 *
 * Construction locates points on Host and owns the resulting flat Host
 * stencils. Device execution uses an explicit copy of data().
 */
class TimePointInterpolator final : public TimeObservationOperator
{
public:
  TimePointInterpolator(Index               num_steps,
                        const MixedFESpace& space,
                        Index               fid,
                        Array<Point3>       pts,
                        Array<Index>        comps,
                        Index               num_prm);

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Index numObservations() const override;

  /** @brief Create an independently owned explicit Device copy. */
  std::unique_ptr<DeviceTimeObservationOperator> copyToDevice(
      CudaContext& ctx) const override;

  void observe(Index             level,
               const HostVector& state,
               const HostVector& prm,
               HostVector&       out) const override;

  void applyStateJac(Index             level,
                     const HostVector& state,
                     const HostVector& prm,
                     const HostVector& dir,
                     HostVector&       out) const override;

  void applyStateJacT(Index             level,
                      const HostVector& state,
                      const HostVector& prm,
                      const HostVector& dir,
                      HostVector&       out) const override;

  void applyParamJac(Index             level,
                     const HostVector& state,
                     const HostVector& prm,
                     const HostVector& dir,
                     HostVector&       out) const override;

  void applyParamJacT(Index             level,
                      const HostVector& state,
                      const HostVector& prm,
                      const HostVector& dir,
                      HostVector&       out) const override;

  /** @brief Flat Host stencils owned by this interpolator. */
  const HostPointInterpolatorData& data() const noexcept;

  const Array<Point3>& pts() const;

  const Array<Index>& comps() const;

  static bool containsPoint(const MixedFESpace& space,
                            Index               fid,
                            const Point3&       point);

  static Array<Point3> filterPointsInside(
      const MixedFESpace&  space,
      Index                fid,
      const Array<Point3>& pts);

private:
  void checkLevel(Index level) const;

  void checkInputs(const HostVector& state,
                   const HostVector& prm) const;

  static HostPointInterpolatorData buildData(
      const MixedFieldView& field,
      Index                 num_states,
      const Array<Point3>&  pts,
      const Array<Index>&   comps);

private:
  Index                     num_steps_{0};
  Index                     num_prm_{0};
  Array<Point3>             pts_;   ///< Observation point coordinates.
  Array<Index>              comps_; ///< Observed components at each point.
  HostPointInterpolatorData data_;  ///< Flat interpolation stencils.
};

/** @brief Explicitly copy the Host data owned by `src` to Device. */
inline void copy(const TimePointInterpolator& src,
                 DevicePointInterpolatorData& dst,
                 CudaContext&                 ctx)
{
  copy(src.data(), dst, ctx);
}

/** @brief Explicitly initialize a Device observation operator from `src`. */
inline void copy(const TimePointInterpolator& src,
                 DeviceTimePointInterpolator& dst,
                 CudaContext&                 ctx)
{
  copy(src.data(), dst.data_, ctx);
  dst.num_steps_ = src.numSteps();
}

} // namespace fem
} // namespace femx
