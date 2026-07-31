#pragma once

#include <utility>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/SystemMatrix.hpp>
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

  /**
   * @brief Return the number of constrained rows.
   */
  Index numBcs() const noexcept
  {
    return constrained_rows_.size();
  }

  /**
   * @brief Return a non-owning view of constrained rows.
   */
  BoundaryMapView<Space> view() const noexcept
  {
    return {constrained_rows_.view()};
  }

private:
  friend void copy(const BoundaryMap<MemorySpace::Host>& source,
                   BoundaryMap<MemorySpace::Device>&     destination,
                   linalg::CudaContext&                  ctx);

  Vector<Space, Index> constrained_rows_; ///< Constrained row indices.
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
 * @param[in]  source - Host boundary map.
 * @param[out] destination - Replaced Device boundary map.
 * @param[in]  ctx - CUDA context used for the asynchronous copy.
 */
void copy(const HostBoundaryMap& source,
          DeviceBoundaryMap&     destination,
          linalg::CudaContext&   ctx);

/**
 * @brief Apply Dirichlet conditions to a Host residual.
 *
 * @param[in]  map - Constrained rows in prescribed-value order.
 * @param[in]  state - Current state.
 * @param[in]  vals - Values prescribed at the constrained rows.
 * @param[out] residual - Residual whose constrained entries are replaced.
 */
void applyDirichletConditions(
    const HostBoundaryMap&     map,
    HostVectorView<const Real> state,
    HostVectorView<const Real> vals,
    HostVectorView<Real>       residual);

/**
 * @brief Apply Dirichlet conditions to an owning Host residual vector.
 *
 * @param[in]  map - Constrained rows in prescribed-value order.
 * @param[in]  state - Current state.
 * @param[in]  vals - Values prescribed at the constrained rows.
 * @param[out] residual - Residual whose constrained entries are replaced.
 */
void applyDirichletConditions(
    const HostBoundaryMap&  map,
    const HostVector<Real>& state,
    const HostVector<Real>& vals,
    HostVector<Real>&       residual);

/**
 * @brief Apply Dirichlet conditions to a Device residual.
 *
 * @param[in]     map - Constrained rows in prescribed-value order.
 * @param[in]     state - Current Device state.
 * @param[in]     vals - Device values prescribed at constrained rows.
 * @param[out]    residual - Device residual whose constrained entries are replaced.
 * @param[in,out] ctx - CUDA context used for the asynchronous update.
 */
void applyDirichletConditions(
    const DeviceBoundaryMap&     map,
    DeviceVectorView<const Real> state,
    DeviceVectorView<const Real> vals,
    DeviceVectorView<Real>       residual,
    linalg::CudaContext&         ctx);

/**
 * @brief Apply Dirichlet conditions to a Jacobian.
 *
 * @param[in]     map - Constrained rows.
 * @param[in,out] jac - Jacobian whose constrained rows become identity rows.
 */
template <MemorySpace Space>
void applyDirichletConditions(
    const BoundaryMap<Space>&    map,
    linalg::SystemMatrix<Space>& jac)
{
  jac.replaceRows(map.view().constrained_rows, 1.0);
}

/**
 * @brief Set constrained Host vector entries to zero.
 */
void zeroBoundary(const HostBoundaryMap& map, HostVectorView<Real> values);

/**
 * @brief Asynchronously set constrained Device vector entries to zero.
 */
void zeroBoundary(const DeviceBoundaryMap& map,
                  DeviceVectorView<Real>   values,
                  linalg::CudaContext&     ctx);

} // namespace femx::assembly
