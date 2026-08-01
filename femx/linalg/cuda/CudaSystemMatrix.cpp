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
    ctx_.matrixHandler().zero(mat_);
  }
}

void CudaSystemMatrix::finalize()
{
}

void CudaSystemMatrix::matvec(DeviceVectorView<const Real> dir,
                              DeviceVector<Real>&          out) const
{
  if (out.size() != mat_.rows())
  {
    out.resize(mat_.rows());
  }
  ctx_.matrixHandler().matvec(mat_, dir, out.view());
}

void CudaSystemMatrix::matvecT(DeviceVectorView<const Real> dir,
                               DeviceVector<Real>&          out) const
{
  if (out.size() != mat_.cols())
  {
    out.resize(mat_.cols());
  }
  ctx_.matrixHandler().matvecT(mat_, dir, out.view());
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

#endif

} // namespace femx::linalg
