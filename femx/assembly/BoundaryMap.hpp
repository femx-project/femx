#pragma once

#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx::assembly
{

/**
 * @brief Non-owning constrained rows in prescribed-value order.
 */
template <MemorySpace Space>
struct BoundaryMapView
{
  VectorView<Space, const Index> constrained_rows;
};

/**
 * @brief Own constrained rows independently of a matrix storage format.
 *
 * The row order defines the order of prescribed boundary values.
 */
template <MemorySpace Space>
class BoundaryMap
{
public:
  BoundaryMap() = default;

  /**
   * @brief Construct a map from validated constrained rows.
   *
   * @param[in] constrained_rows - Unique nonnegative row indices.
   */
  explicit BoundaryMap(Vector<Space, Index> constrained_rows)
    : constrained_rows_(std::move(constrained_rows))
  {
  }

  /** @brief Return the number of constrained rows. */
  Index numBcs() const noexcept
  {
    return constrained_rows_.size();
  }

  /** @brief Return a non-owning view of constrained rows. */
  BoundaryMapView<Space> view() const noexcept
  {
    return {constrained_rows_.view()};
  }

private:
  friend void copy(const BoundaryMap<MemorySpace::Host>& source,
                   BoundaryMap<MemorySpace::Device>&     destination,
                   linalg::CudaContext&                  ctx);

  Vector<Space, Index> constrained_rows_;
};

using HostBoundaryMap   = BoundaryMap<MemorySpace::Host>;
using DeviceBoundaryMap = BoundaryMap<MemorySpace::Device>;

/**
 * @brief Validate and own constrained Host rows.
 *
 * @param[in] rows - Unique nonnegative constrained rows.
 */
HostBoundaryMap makeBoundaryMap(const HostVector<Index>& rows);

/**
 * @brief Copy constrained rows to Device storage.
 *
 * @param[in] source - Host boundary map.
 * @param[out] destination - Replaced Device boundary map.
 * @param[in] ctx - CUDA context used for the asynchronous copy.
 */
void copy(const HostBoundaryMap& source,
          DeviceBoundaryMap&     destination,
          linalg::CudaContext&   ctx);

/**
 * @brief Replace constrained residual entries by state minus prescribed value.
 */
void replaceRes(const HostBoundaryMap&     map,
                HostVectorView<const Real> state,
                HostVectorView<const Real> prescribed_values,
                HostVectorView<Real>       residual);

/** @brief Owning-vector convenience overload of replaceRes(). */
void replaceRes(const HostBoundaryMap&  map,
                const HostVector<Real>& state,
                const HostVector<Real>& prescribed_values,
                HostVector<Real>&       residual);

/**
 * @brief Asynchronously replace constrained Device residual entries.
 */
void replaceRes(const DeviceBoundaryMap&     map,
                DeviceVectorView<const Real> state,
                DeviceVectorView<const Real> prescribed_values,
                DeviceVectorView<Real>       residual,
                linalg::CudaContext&         ctx);

/** @brief Set constrained Host vector entries to zero. */
void zeroBoundary(const HostBoundaryMap& map, HostVectorView<Real> values);

/** @brief Asynchronously set constrained Device vector entries to zero. */
void zeroBoundary(const DeviceBoundaryMap& map,
                  DeviceVectorView<Real>   values,
                  linalg::CudaContext&     ctx);

} // namespace femx::assembly
