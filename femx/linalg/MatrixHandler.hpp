#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

/**
 * @brief Define matrix operations for one memory space.
 *
 * @tparam Space - Storage location on which the operations act.
 */
template <MemorySpace Space>
class MatrixHandler;

/**
 * @brief Define backend-independent operations on Host matrices.
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
   * @param[in]  src - Source matrix.
   * @param[out] dst - Transposed destination.
   * @throws std::runtime_error If validation fails.
   */
  virtual void transpose(const HostCsrMatrix& src,
                         HostCsrMatrix&       dst) const = 0;

  /**
   * @brief Compute `out = alpha * mat * dir + beta * out`.
   *
   * @param[in]     mat   - Host CSR matrix with shape `(m, n)`.
   * @param[in]     dir   - Input vector containing `n` values.
   * @param[in,out] out   - Output vector containing `m` values.
   * @param[in]     alpha - Scale applied to the matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvec(const HostCsrMatrix&       mat,
                      HostVectorView<const Real> dir,
                      HostVectorView<Real>       out,
                      Real                       alpha = 1.0,
                      Real                       beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * transpose(mat) * dir + beta * out`.
   *
   * @param[in]     mat   - Host CSR matrix with shape `(m, n)`.
   * @param[in]     dir   - Input vector containing `m` values.
   * @param[in,out] out   - Output vector containing `n` values.
   * @param[in]     alpha - Scale applied to the transposed matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvecT(const HostCsrMatrix&       mat,
                       HostVectorView<const Real> dir,
                       HostVectorView<Real>       out,
                       Real                       alpha = 1.0,
                       Real                       beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * mat * dir + beta * out` for a dense matrix.
   *
   * @param[in]     mat   - Dense Host matrix with shape `(m, n)`.
   * @param[in]     dir   - Input vector containing `n` values.
   * @param[in,out] out   - Output vector containing `m` values.
   * @param[in]     alpha - Scale applied to the matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvec(HostMatrixView<const Real> mat,
                      HostVectorView<const Real> dir,
                      HostVectorView<Real>       out,
                      Real                       alpha = 1.0,
                      Real                       beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * transpose(mat) * dir + beta * out`.
   *
   * @param[in]     mat   - Dense Host matrix with shape `(m, n)`.
   * @param[in]     dir   - Input vector containing `m` values.
   * @param[in,out] out   - Output vector containing `n` values.
   * @param[in]     alpha - Scale applied to the transposed matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvecT(HostMatrixView<const Real> mat,
                       HostVectorView<const Real> dir,
                       HostVectorView<Real>       out,
                       Real                       alpha = 1.0,
                       Real                       beta  = 0.0) const = 0;
};

/**
 * @brief Define backend-independent operations on Device matrices.
 */
template <>
class MatrixHandler<MemorySpace::Device>
{
public:
  virtual ~MatrixHandler() = default;

  /**
   * @brief Set every numeric value in a Device CSR matrix to zero.
   *
   * @param[in,out] mat - Matrix whose Device values are cleared.
   * @throws std::runtime_error If validation fails.
   */
  virtual void zero(DeviceCsrMatrix& mat) const = 0;

  /**
   * @brief Construct or update the explicit transpose of a Device CSR matrix.
   *
   * @param[in]  src - Source Device CSR matrix.
   * @param[out] dst - Transposed Device destination.
   * @throws std::runtime_error If validation fails.
   */
  virtual void transpose(const DeviceCsrMatrix& src,
                         DeviceCsrMatrix&       dst) const = 0;

  /**
   * @brief Compute `out = alpha * mat * dir + beta * out`.
   *
   * @param[in]     mat   - Device CSR matrix with shape `(m, n)`.
   * @param[in]     dir   - Device input vector containing `n` values.
   * @param[in,out] out   - Device output vector containing `m` values.
   * @param[in]     alpha - Scale applied to the matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvec(const DeviceCsrMatrix&       mat,
                      DeviceVectorView<const Real> dir,
                      DeviceVectorView<Real>       out,
                      Real                         alpha = 1.0,
                      Real                         beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * transpose(mat) * dir + beta * out`.
   *
   * @param[in]     mat   - Device CSR matrix with shape `(m, n)`.
   * @param[in]     dir   - Device input vector containing `m` values.
   * @param[in,out] out   - Device output vector containing `n` values.
   * @param[in]     alpha - Scale applied to the transposed matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvecT(const DeviceCsrMatrix&       mat,
                       DeviceVectorView<const Real> dir,
                       DeviceVectorView<Real>       out,
                       Real                         alpha = 1.0,
                       Real                         beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * mat * dir + beta * out` for a dense matrix.
   *
   * @param[in]     mat   - Dense Device matrix with shape `(m, n)`.
   * @param[in]     dir   - Device input vector containing `n` values.
   * @param[in,out] out   - Device output vector containing `m` values.
   * @param[in]     alpha - Scale applied to the matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvec(DeviceMatrixView<const Real> mat,
                      DeviceVectorView<const Real> dir,
                      DeviceVectorView<Real>       out,
                      Real                         alpha = 1.0,
                      Real                         beta  = 0.0) const = 0;

  /**
   * @brief Compute `out = alpha * transpose(mat) * dir + beta * out`.
   *
   * @param[in]     mat   - Dense Device matrix with shape `(m, n)`.
   * @param[in]     dir   - Device input vector containing `m` values.
   * @param[in,out] out   - Device output vector containing `n` values.
   * @param[in]     alpha - Scale applied to the transposed matrix-vector product.
   * @param[in]     beta  - Scale applied to the previous output values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void matvecT(DeviceMatrixView<const Real> mat,
                       DeviceVectorView<const Real> dir,
                       DeviceVectorView<Real>       out,
                       Real                         alpha = 1.0,
                       Real                         beta  = 0.0) const = 0;
};

} // namespace femx::linalg
