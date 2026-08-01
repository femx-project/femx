#pragma once

#include <femx/linalg/MatrixHandler.hpp>

namespace femx::linalg
{

class CudaContext;

/**
 * @brief Implement Device CSR and dense matrix operations with CUDA.
 */
class CudaMatrixHandler final : public MatrixHandler<MemorySpace::Device>
{
  using Base = MatrixHandler<MemorySpace::Device>;

public:
  explicit CudaMatrixHandler(CudaContext& ctx) noexcept;

  /**
   * @copydoc Base::zero()
   */
  void zero(DeviceCsrMatrix& mat) const override;

  /**
   * @copydoc Base::transpose()
   */
  void transpose(const DeviceCsrMatrix& src,
                 DeviceCsrMatrix&       dst) const override;

  /**
   * @copydoc Base::matvec(const DeviceCsrMatrix&,DeviceVectorView<const Real>,DeviceVectorView<Real>,Real,Real) const
   */
  void matvec(const DeviceCsrMatrix&       mat,
              DeviceVectorView<const Real> dir,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const override;

  /**
   * @copydoc Base::matvecT(const DeviceCsrMatrix&,DeviceVectorView<const Real>,DeviceVectorView<Real>,Real,Real) const
   */
  void matvecT(const DeviceCsrMatrix&       mat,
               DeviceVectorView<const Real> dir,
               DeviceVectorView<Real>       out,
               Real                         alpha = 1.0,
               Real                         beta  = 0.0) const override;

  /**
   * @copydoc Base::matvec(DeviceMatrixView<const Real>,DeviceVectorView<const Real>,DeviceVectorView<Real>,Real,Real) const
   */
  void matvec(DeviceMatrixView<const Real> mat,
              DeviceVectorView<const Real> dir,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const override;

  /**
   * @copydoc Base::matvecT(DeviceMatrixView<const Real>,DeviceVectorView<const Real>,DeviceVectorView<Real>,Real,Real) const
   */
  void matvecT(DeviceMatrixView<const Real> mat,
               DeviceVectorView<const Real> dir,
               DeviceVectorView<Real>       out,
               Real                         alpha = 1.0,
               Real                         beta  = 0.0) const override;

private:
  CudaContext& ctx_; ///< Bound CUDA execution context.
};

} // namespace femx::linalg
