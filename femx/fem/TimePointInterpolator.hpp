#pragma once

#include <femx/common/Context.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>

namespace femx
{
namespace fem
{

using inverse::DeviceTimeObservationOperator;
using inverse::TimeObservationOperator;

template <MemorySpace Space>
class PointInterpolatorData;

class TimePointInterpolator;
class DeviceTimePointInterpolator;

using HostPointInterpolatorData =
    PointInterpolatorData<MemorySpace::Host>;
using DevicePointInterpolatorData =
    PointInterpolatorData<MemorySpace::Device>;

/** @brief Memory-space owner of the reusable observation CSR matrix. */
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
    return mat_.cols();
  }

  /** @brief Number of point-component observations. */
  Index numObservations() const noexcept
  {
    return mat_.rows();
  }

  /** @brief Number of flattened interpolation entries. */
  Index numEntries() const noexcept
  {
    return mat_.nnz();
  }

  /** @brief Return the interpolation matrix `H`. */
  const CsrMatrix<Space>& matrix() const noexcept
  {
    return mat_;
  }

private:
  friend class TimePointInterpolator;

  friend void copy(const HostPointInterpolatorData& src,
                   DevicePointInterpolatorData&     dst,
                   CudaContext&                     ctx);

  CsrMatrix<Space> mat_;
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
  linalg::CudaMatrixHandler mat_handler(ctx);
  DeviceCsrPattern          pattern;
  femx::copy(src.mat_.pattern(), pattern, ctx);
  DeviceCsrMatrix mat(pattern);
  mat_handler.copy(src.mat_, mat);
  dst.mat_ = std::move(mat);
}

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
