#pragma once

#include <femx/linalg/MatrixHandler.hpp>

namespace femx::linalg
{

/**
 * @brief Implement serial Host CSR and dense matrix operations.
 */
class HostMatrixHandler final : public MatrixHandler<MemorySpace::Host>
{
public:
  void zero(HostCsrMatrix& mat) const override;

  void transpose(const HostCsrMatrix& src,
                 HostCsrMatrix&       dst) const override;

  void matvec(const HostCsrMatrix&       mat,
              HostVectorView<const Real> dir,
              HostVectorView<Real>       out,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const override;

  void matvecT(const HostCsrMatrix&       mat,
               HostVectorView<const Real> dir,
               HostVectorView<Real>       out,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0) const override;

  void matvec(HostMatrixView<const Real> mat,
              HostVectorView<const Real> dir,
              HostVectorView<Real>       out,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const override;

  void matvecT(HostMatrixView<const Real> mat,
               HostVectorView<const Real> dir,
               HostVectorView<Real>       out,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0) const override;
};

} // namespace femx::linalg
