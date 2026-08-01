#pragma once

#include <femx/linalg/MatrixHandler.hpp>

namespace femx::linalg
{

/**
 * @brief Implement serial Host CSR and dense matrix operations.
 */
class HostMatrixHandler final : public MatrixHandler<MemorySpace::Host>
{
  using Base = MatrixHandler<MemorySpace::Host>;

public:
  /**
   * @copydoc Base::zero()
   */
  void zero(HostCsrMatrix& mat) const override;

  /**
   * @copydoc Base::transpose()
   */
  void transpose(const HostCsrMatrix& src,
                 HostCsrMatrix&       dst) const override;

  /**
   * @copydoc Base::matvec(const HostCsrMatrix&,HostVectorView<const Real>,HostVectorView<Real>,Real,Real) const
   */
  void matvec(const HostCsrMatrix&       mat,
              HostVectorView<const Real> dir,
              HostVectorView<Real>       out,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const override;

  /**
   * @copydoc Base::matvecT(const HostCsrMatrix&,HostVectorView<const Real>,HostVectorView<Real>,Real,Real) const
   */
  void matvecT(const HostCsrMatrix&       mat,
               HostVectorView<const Real> dir,
               HostVectorView<Real>       out,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0) const override;

  /**
   * @copydoc Base::matvec(HostMatrixView<const Real>,HostVectorView<const Real>,HostVectorView<Real>,Real,Real) const
   */
  void matvec(HostMatrixView<const Real> mat,
              HostVectorView<const Real> dir,
              HostVectorView<Real>       out,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const override;

  /**
   * @copydoc Base::matvecT(HostMatrixView<const Real>,HostVectorView<const Real>,HostVectorView<Real>,Real,Real) const
   */
  void matvecT(HostMatrixView<const Real> mat,
               HostVectorView<const Real> dir,
               HostVectorView<Real>       out,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0) const override;
};

} // namespace femx::linalg
