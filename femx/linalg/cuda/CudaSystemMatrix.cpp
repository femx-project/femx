#include <stdexcept>

#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx::linalg
{

CudaSystemMatrix::CudaSystemMatrix(CudaContext& ctx) noexcept
  : ctx_(ctx)
{
}

void CudaSystemMatrix::setup(const HostCsrPattern& pattern)
{
  if (matrix_.pattern().layoutId() != pattern.layoutId())
  {
    DeviceCsrPattern d_pattern;
    femx::copy(pattern, d_pattern, ctx_);
    matrix_      = DeviceCsrMatrix(d_pattern);
    constraints_ = {};
  }
  else
  {
    ctx_.vectorHandler().zero(matrix_.vals().view());
  }
}

void CudaSystemMatrix::finalize()
{
}

void CudaSystemMatrix::apply(DeviceVectorView<const Real> direction,
                             DeviceVector<Real>&          out) const
{
  if (out.size() != matrix_.rows())
  {
    out.resize(matrix_.rows());
  }
  apply(matrix_, direction, out.view());
}

void CudaSystemMatrix::applyT(DeviceVectorView<const Real> direction,
                              DeviceVector<Real>&          out) const
{
  if (out.size() != matrix_.cols())
  {
    out.resize(matrix_.cols());
  }
  applyT(matrix_, direction, out.view());
}

DeviceCsrAssemblyView CudaSystemMatrix::assemblyView() noexcept
{
  return {matrix_.rows(),
          matrix_.cols(),
          matrix_.nnz(),
          matrix_.rowPtrData(),
          matrix_.colIndData(),
          matrix_.vals().view()};
}

const DeviceCsrMatrix& CudaSystemMatrix::matrix() const noexcept
{
  return matrix_;
}

#if !defined(FEMX_HAS_CUDA)
namespace
{
[[noreturn]] void cudaUnavailable()
{
  throw std::runtime_error(
      "CudaSystemMatrix operations require FEMX_ENABLE_CUDA");
}
} // namespace

void CudaSystemMatrix::replaceRows(DeviceVectorView<const Index>, Real)
{
  cudaUnavailable();
}

void CudaSystemMatrix::eliminateColumns(DeviceVectorView<const Index>,
                                        DeviceVectorView<const Real>,
                                        DeviceVectorView<Real>)
{
  cudaUnavailable();
}

void CudaSystemMatrix::ensureConstraints(DeviceVectorView<const Index>)
{
  cudaUnavailable();
}

void CudaSystemMatrix::transpose(const DeviceCsrMatrix&,
                                 DeviceCsrMatrix&) const
{
  cudaUnavailable();
}

void CudaSystemMatrix::apply(const DeviceCsrMatrix&,
                             DeviceVectorView<const Real>,
                             DeviceVectorView<Real>,
                             Real,
                             Real) const
{
  cudaUnavailable();
}

void CudaSystemMatrix::applyT(const DeviceCsrMatrix&,
                              DeviceVectorView<const Real>,
                              DeviceVectorView<Real>,
                              Real,
                              Real) const
{
  cudaUnavailable();
}

void CudaSystemMatrix::apply(DeviceMatrixView<const Real>,
                             DeviceVectorView<const Real>,
                             DeviceVectorView<Real>,
                             Real,
                             Real) const
{
  cudaUnavailable();
}

void CudaSystemMatrix::applyT(DeviceMatrixView<const Real>,
                              DeviceVectorView<const Real>,
                              DeviceVectorView<Real>,
                              Real,
                              Real) const
{
  cudaUnavailable();
}
#endif

} // namespace femx::linalg
