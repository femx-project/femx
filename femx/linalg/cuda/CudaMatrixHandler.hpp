#pragma once

#include <femx/linalg/MatrixHandler.hpp>

namespace femx::linalg
{

class CudaContext;

/**
 * @brief Implement Device CSR and dense matrix operations with CUDA.
 *
 * Operations are enqueued on the stream owned by the bound context.
 */
class CudaMatrixHandler final : public MatrixHandler<MemorySpace::Device>
{
public:
  explicit CudaMatrixHandler(CudaContext& ctx) noexcept;

  void zero(DeviceCsrMatrix& mat) const override;

  void transpose(const DeviceCsrMatrix& src,
                 DeviceCsrMatrix&       dst) const override;

  void matvec(const DeviceCsrMatrix&       mat,
              DeviceVectorView<const Real> dir,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const override;

  void matvecT(const DeviceCsrMatrix&       mat,
               DeviceVectorView<const Real> dir,
               DeviceVectorView<Real>       out,
               Real                         alpha = 1.0,
               Real                         beta  = 0.0) const override;

  void matvec(DeviceMatrixView<const Real> mat,
              DeviceVectorView<const Real> dir,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const override;

  void matvecT(DeviceMatrixView<const Real> mat,
               DeviceVectorView<const Real> dir,
               DeviceVectorView<Real>       out,
               Real                         alpha = 1.0,
               Real                         beta  = 0.0) const override;

private:
  CudaContext& ctx_; ///< Bound CUDA execution context.
};

} // namespace femx::linalg
