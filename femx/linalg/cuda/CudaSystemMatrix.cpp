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
  if (mat_.pattern().layoutId() != pattern.layoutId())
  {
    DeviceCsrPattern d_pattern;
    femx::copy(pattern, d_pattern, ctx_);
    mat_         = DeviceCsrMatrix(d_pattern);
    constraints_ = {};
  }
  else
  {
    ctx_.vectorHandler().zero(mat_.vals().view());
  }
}

void CudaSystemMatrix::finalize()
{
}

void CudaSystemMatrix::apply(DeviceVectorView<const Real> dir,
                             DeviceVector<Real>&          out) const
{
  if (out.size() != mat_.rows())
  {
    out.resize(mat_.rows());
  }
  apply(mat_, dir, out.view());
}

void CudaSystemMatrix::applyT(DeviceVectorView<const Real> dir,
                              DeviceVector<Real>&          out) const
{
  if (out.size() != mat_.cols())
  {
    out.resize(mat_.cols());
  }
  applyT(mat_, dir, out.view());
}

DeviceCsrAssemblyView CudaSystemMatrix::assemblyView() noexcept
{
  return {mat_.rows(),
          mat_.cols(),
          mat_.nnz(),
          mat_.rowPtrData(),
          mat_.colIndData(),
          mat_.vals().view()};
}

const DeviceCsrMatrix& CudaSystemMatrix::matrix() const noexcept
{
  return mat_;
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
