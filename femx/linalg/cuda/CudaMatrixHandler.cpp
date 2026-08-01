#include <stdexcept>

#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaMatrixHandler.hpp>

namespace femx::linalg
{

CudaMatrixHandler::CudaMatrixHandler(CudaContext& ctx) noexcept
  : ctx_(ctx)
{
}

void CudaMatrixHandler::zero(DeviceCsrMatrix& mat) const
{
  ctx_.vectorHandler().zero(mat.vals().view());
}

#if !defined(FEMX_HAS_CUDA)
namespace
{
[[noreturn]] void cudaUnavailable()
{
  throw std::runtime_error(
      "CudaMatrixHandler operations require FEMX_ENABLE_CUDA");
}
} // namespace

void CudaMatrixHandler::transpose(const DeviceCsrMatrix&,
                                  DeviceCsrMatrix&) const
{
  cudaUnavailable();
}

void CudaMatrixHandler::matvec(const DeviceCsrMatrix&,
                               DeviceVectorView<const Real>,
                               DeviceVectorView<Real>,
                               Real,
                               Real) const
{
  cudaUnavailable();
}

void CudaMatrixHandler::matvecT(const DeviceCsrMatrix&,
                                DeviceVectorView<const Real>,
                                DeviceVectorView<Real>,
                                Real,
                                Real) const
{
  cudaUnavailable();
}

void CudaMatrixHandler::matvec(DeviceMatrixView<const Real>,
                               DeviceVectorView<const Real>,
                               DeviceVectorView<Real>,
                               Real,
                               Real) const
{
  cudaUnavailable();
}

void CudaMatrixHandler::matvecT(DeviceMatrixView<const Real>,
                                DeviceVectorView<const Real>,
                                DeviceVectorView<Real>,
                                Real,
                                Real) const
{
  cudaUnavailable();
}
#endif

} // namespace femx::linalg
