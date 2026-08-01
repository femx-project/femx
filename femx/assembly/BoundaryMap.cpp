#include <stdexcept>

#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Checks.hpp>

namespace femx::assembly
{

HostBoundaryMap makeBoundaryMap(const HostVector<Index>& rows)
{
  HostVector<Index> constrained_rows(rows);
  for (Index i = 0; i < constrained_rows.size(); ++i)
  {
    require(constrained_rows[i] >= 0,
            "BoundaryMap constrained rows must be nonnegative");
    for (Index previous = 0; previous < i; ++previous)
    {
      require(constrained_rows[i] != constrained_rows[previous],
              "BoundaryMap constrained rows must be unique");
    }
  }
  return HostBoundaryMap(std::move(constrained_rows));
}

void copy(const HostBoundaryMap&                source,
          DeviceBoundaryMap&                    destination,
          linalg::Context<MemorySpace::Device>& ctx)
{
  DeviceVector<Index> constrained_rows;
  ctx.vectorHandler().copy(source.constrained_rows_, constrained_rows);
  destination = DeviceBoundaryMap(std::move(constrained_rows));
}

void applyDirichletConditions(
    const HostBoundaryMap&     map,
    HostVectorView<const Real> state,
    HostVectorView<const Real> vals,
    HostVectorView<Real>       residual)
{
  const auto rows = map.view().constrained_rows;
  require(vals.size() == rows.size(),
          "BoundaryMap prescribed-value size mismatch");
  require(!detail::overlaps(state, residual)
              && !detail::overlaps(vals, residual),
          "BoundaryMap residual output must not alias its inputs");
  for (Index i = 0; i < rows.size(); ++i)
  {
    require(rows[i] < state.size() && rows[i] < residual.size(),
            "BoundaryMap constrained row is out of vector range");
    residual[rows[i]] = state[rows[i]] - vals[i];
  }
}

void applyDirichletConditions(
    const HostBoundaryMap&  map,
    const HostVector<Real>& state,
    const HostVector<Real>& vals,
    HostVector<Real>&       residual)
{
  applyDirichletConditions(
      map, state.view(), vals.view(), residual.view());
}

void zeroBoundary(const HostBoundaryMap& map,
                  HostVectorView<Real>   values)
{
  const auto rows = map.view().constrained_rows;
  for (Index row : rows)
  {
    require(row < values.size(),
            "BoundaryMap constrained row is out of vector range");
    values[row] = 0.0;
  }
}

#if !defined(FEMX_HAS_CUDA)
void applyDirichletConditions(const DeviceBoundaryMap&,
                              DeviceVectorView<const Real>,
                              DeviceVectorView<const Real>,
                              DeviceVectorView<Real>,
                              linalg::CudaContext&)
{
  throw std::runtime_error(
      "BoundaryMap CUDA operations require FEMX_ENABLE_CUDA");
}

void zeroBoundary(const DeviceBoundaryMap&,
                  DeviceVectorView<Real>,
                  linalg::CudaContext&)
{
  throw std::runtime_error(
      "BoundaryMap CUDA operations require FEMX_ENABLE_CUDA");
}
#endif

} // namespace femx::assembly
