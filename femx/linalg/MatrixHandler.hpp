#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

/**
 * @brief Define matrix operations for one memory space.
 */
template <MemorySpace Space>
class MatrixHandler;

/**
 * @brief Define backend-independent Host matrix operations.
 */
template <>
class MatrixHandler<MemorySpace::Host>
{
public:
  virtual ~MatrixHandler() = default;

  /**
   * @brief Set every numeric value in a Host CSR matrix to zero.
   *
   * @param[in,out] mat - Matrix whose values are cleared.
   */
  virtual void zero(HostCsrMatrix& mat) const = 0;

  /**
   * @brief Construct the explicit transpose of a Host CSR matrix.
   *
   * @param[in] src - Source matrix.
   * @param[out] dst - Transposed destination.
   */
  virtual void transpose(const HostCsrMatrix& src,
                         HostCsrMatrix&       dst) const = 0;

  /**
   * @brief Compute `out = alpha * mat * dir + beta * out`.
   */
  virtual void matvec(const HostCsrMatrix&       mat,
                      HostVectorView<const Real> dir,
                      HostVectorView<Real>       out,
                      Real                       alpha = 1.0,
                      Real                       beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * transpose(mat) * dir + beta * out`.
   */
  virtual void matvecT(const HostCsrMatrix&       mat,
                       HostVectorView<const Real> dir,
                       HostVectorView<Real>       out,
                       Real                       alpha = 1.0,
                       Real                       beta  = 0.0) const = 0;

  /**
   * @brief Compute a row-major dense Host matrix-vector product.
   */
  virtual void matvec(HostMatrixView<const Real> mat,
                      HostVectorView<const Real> dir,
                      HostVectorView<Real>       out,
                      Real                       alpha = 1.0,
                      Real                       beta  = 0.0) const = 0;

  /**
   * @brief Compute a transposed row-major dense Host matrix-vector product.
   */
  virtual void matvecT(HostMatrixView<const Real> mat,
                       HostVectorView<const Real> dir,
                       HostVectorView<Real>       out,
                       Real                       alpha = 1.0,
                       Real                       beta  = 0.0) const = 0;
};

/**
 * @brief Define backend-independent Device matrix operations.
 */
template <>
class MatrixHandler<MemorySpace::Device>
{
public:
  virtual ~MatrixHandler() = default;

  /**
   * @brief Set every numeric value in a Device CSR matrix to zero.
   */
  virtual void zero(DeviceCsrMatrix& mat) const = 0;

  /**
   * @brief Construct or update the explicit transpose of a Device CSR matrix.
   */
  virtual void transpose(const DeviceCsrMatrix& src,
                         DeviceCsrMatrix&       dst) const = 0;

  /**
   * @brief Compute `out = alpha * mat * dir + beta * out`.
   */
  virtual void matvec(const DeviceCsrMatrix&       mat,
                      DeviceVectorView<const Real> dir,
                      DeviceVectorView<Real>       out,
                      Real                         alpha = 1.0,
                      Real                         beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * transpose(mat) * dir + beta * out`.
   */
  virtual void matvecT(const DeviceCsrMatrix&       mat,
                       DeviceVectorView<const Real> dir,
                       DeviceVectorView<Real>       out,
                       Real                         alpha = 1.0,
                       Real                         beta  = 0.0) const = 0;

  /**
   * @brief Compute a row-major dense Device matrix-vector product.
   */
  virtual void matvec(DeviceMatrixView<const Real> mat,
                      DeviceVectorView<const Real> dir,
                      DeviceVectorView<Real>       out,
                      Real                         alpha = 1.0,
                      Real                         beta  = 0.0) const = 0;

  /**
   * @brief Compute a transposed row-major dense Device matrix-vector product.
   */
  virtual void matvecT(DeviceMatrixView<const Real> mat,
                       DeviceVectorView<const Real> dir,
                       DeviceVectorView<Real>       out,
                       Real                         alpha = 1.0,
                       Real                         beta  = 0.0) const = 0;
};

} // namespace femx::linalg
