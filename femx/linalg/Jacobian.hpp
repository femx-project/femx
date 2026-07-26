#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>

namespace femx::linalg
{

/**
 * @brief Non-owning Host element contribution passed to a Jacobian.
 *
 * The values are row-major. CSR entries correspond one-to-one with values and
 * are used only by CSR-backed implementations.
 */
struct ElementJacobianView
{
  HostVectorView<const Index> rows;
  HostVectorView<const Index> columns;
  HostVectorView<const Index> csr_entries;
  HostMatrixView<const Real>  values;
};

/**
 * @brief Non-owning Device CSR storage used by assembly kernels.
 */
struct DeviceCsrAssemblyView
{
  Index                  rows{0};
  Index                  columns{0};
  Index                  nonzeros{0};
  const Index*           row_offsets{nullptr};
  const Index*           column_indices{nullptr};
  DeviceVectorView<Real> values;
};

/** @brief Define Jacobian assembly and application in one memory space. */
template <MemorySpace Space>
class Jacobian;

/** @brief Define Host Jacobian assembly and application. */
template <>
class Jacobian<MemorySpace::Host>
{
public:
  virtual ~Jacobian() = default;

  /**
   * @brief Prepare zero-valued storage for a Host CSR pattern.
   *
   * @param[in] pattern - Canonical global sparsity pattern.
   */
  virtual void begin(const HostCsrPattern& pattern) = 0;

  /**
   * @brief Add one element contribution.
   *
   * @param[in] element - Element rows, columns, CSR entries, and values.
   */
  virtual void addElement(const ElementJacobianView& element) = 0;

  /**
   * @brief Replace constrained rows by diagonal rows.
   *
   * @param[in] rows - Constrained global row indices.
   * @param[in] diagonal - Replacement diagonal value.
   */
  virtual void replaceRows(HostVectorView<const Index> rows,
                           Real                        diagonal) = 0;

  /**
   * @brief Eliminate constrained columns and correct a right-hand side.
   *
   * The value order must match `rows`.
   *
   * @param[in] rows - Constrained global row indices.
   * @param[in] values - Prescribed values.
   * @param[in,out] rhs - Right-hand side corrected in place.
   */
  virtual void eliminateColumns(HostVectorView<const Index> rows,
                                HostVectorView<const Real>  values,
                                HostVectorView<Real>        rhs) = 0;

  /** @brief Complete assembly before application or solution. */
  virtual void finalize() = 0;

  /**
   * @brief Compute the Jacobian product.
   *
   * @param[in] direction - Input direction.
   * @param[out] out - Resized output vector.
   */
  virtual void apply(HostVectorView<const Real> direction,
                     HostVector<Real>&          out) const = 0;

  /**
   * @brief Compute the transposed Jacobian product.
   *
   * @param[in] direction - Input direction.
   * @param[out] out - Resized output vector.
   */
  virtual void applyT(HostVectorView<const Real> direction,
                      HostVector<Real>&          out) const = 0;
};

/** @brief Define Device Jacobian assembly and application. */
template <>
class Jacobian<MemorySpace::Device>
{
public:
  virtual ~Jacobian() = default;

  /**
   * @brief Prepare zero-valued Device storage from a Host CSR pattern.
   *
   * @param[in] pattern - Canonical global sparsity pattern.
   */
  virtual void begin(const HostCsrPattern& pattern) = 0;

  /**
   * @brief Replace constrained rows by diagonal rows.
   *
   * @param[in] rows - Device constrained row indices.
   * @param[in] diagonal - Replacement diagonal value.
   */
  virtual void replaceRows(DeviceVectorView<const Index> rows,
                           Real                          diagonal) = 0;

  /**
   * @brief Eliminate constrained columns and correct a right-hand side.
   *
   * @param[in] rows - Device constrained row indices.
   * @param[in] values - Device prescribed values.
   * @param[in,out] rhs - Device right-hand side corrected in place.
   */
  virtual void eliminateColumns(DeviceVectorView<const Index> rows,
                                DeviceVectorView<const Real>  values,
                                DeviceVectorView<Real>        rhs) = 0;

  /** @brief Complete assembly before application or solution. */
  virtual void finalize() = 0;

  /**
   * @brief Compute the Device Jacobian product.
   *
   * @param[in] direction - Device input direction.
   * @param[out] out - Resized Device output vector.
   */
  virtual void apply(DeviceVectorView<const Real> direction,
                     DeviceVector<Real>&          out) const = 0;

  /**
   * @brief Compute the transposed Device Jacobian product.
   *
   * @param[in] direction - Device input direction.
   * @param[out] out - Resized Device output vector.
   */
  virtual void applyT(DeviceVectorView<const Real> direction,
                      DeviceVector<Real>&          out) const = 0;
};

} // namespace femx::linalg
