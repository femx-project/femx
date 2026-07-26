#include <stdexcept>

#include <femx/linalg/cuda/CudaJacobian.hpp>

namespace femx::linalg
{

CudaJacobian::CudaJacobian(CudaContext& ctx) noexcept
  : ctx_(ctx)
{
}

void CudaJacobian::begin(const HostCsrPattern& pattern)
{
  if (matrix_.pattern().layoutId() != pattern.layoutId())
  {
    DeviceCsrPattern device_pattern;
    femx::copy(pattern, device_pattern, ctx_);
    matrix_      = DeviceCsrMatrix(device_pattern);
    constraints_ = {};
  }
  else
  {
    ctx_.vectors().zero(matrix_.vals().view());
  }
}

void CudaJacobian::finalize()
{
}

void CudaJacobian::apply(DeviceVectorView<const Real> direction,
                         DeviceVector<Real>&          out) const
{
  if (out.size() != matrix_.rows())
  {
    out.resize(matrix_.rows());
  }
  apply(matrix_, direction, out.view());
}

void CudaJacobian::applyT(DeviceVectorView<const Real> direction,
                          DeviceVector<Real>&          out) const
{
  if (out.size() != matrix_.cols())
  {
    out.resize(matrix_.cols());
  }
  applyT(matrix_, direction, out.view());
}

DeviceCsrAssemblyView CudaJacobian::assemblyView() noexcept
{
  return {matrix_.rows(),
          matrix_.cols(),
          matrix_.nnz(),
          matrix_.rowPtrData(),
          matrix_.colIndData(),
          matrix_.vals().view()};
}

const DeviceCsrMatrix& CudaJacobian::matrix() const noexcept
{
  return matrix_;
}

#if !defined(FEMX_HAS_CUDA)
namespace
{
[[noreturn]] void cudaUnavailable()
{
  throw std::runtime_error(
      "CudaJacobian operations require FEMX_ENABLE_CUDA");
}
} // namespace

void CudaJacobian::replaceRows(DeviceVectorView<const Index>, Real)
{
  cudaUnavailable();
}

void CudaJacobian::eliminateColumns(DeviceVectorView<const Index>,
                                    DeviceVectorView<const Real>,
                                    DeviceVectorView<Real>)
{
  cudaUnavailable();
}

void CudaJacobian::ensureConstraints(DeviceVectorView<const Index>)
{
  cudaUnavailable();
}

void CudaJacobian::transpose(const DeviceCsrMatrix&,
                             DeviceCsrMatrix&) const
{
  cudaUnavailable();
}

void CudaJacobian::apply(const DeviceCsrMatrix&,
                         DeviceVectorView<const Real>,
                         DeviceVectorView<Real>,
                         Real,
                         Real) const
{
  cudaUnavailable();
}

void CudaJacobian::applyT(const DeviceCsrMatrix&,
                          DeviceVectorView<const Real>,
                          DeviceVectorView<Real>,
                          Real,
                          Real) const
{
  cudaUnavailable();
}

void CudaJacobian::apply(DeviceMatrixView<const Real>,
                         DeviceVectorView<const Real>,
                         DeviceVectorView<Real>,
                         Real,
                         Real) const
{
  cudaUnavailable();
}

void CudaJacobian::applyT(DeviceMatrixView<const Real>,
                          DeviceVectorView<const Real>,
                          DeviceVectorView<Real>,
                          Real,
                          Real) const
{
  cudaUnavailable();
}
#endif

} // namespace femx::linalg
