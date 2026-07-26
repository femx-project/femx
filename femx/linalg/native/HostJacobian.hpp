#pragma once

#include <memory>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Jacobian.hpp>

namespace femx::linalg
{

/**
 * @brief Own and operate on a Host CSR Jacobian.
 *
 * Constraint metadata derived from the current pattern and constrained rows
 * is cached by this object.
 */
class HostJacobian final : public Jacobian<MemorySpace::Host>
{
public:
  /**
   * @brief Bind the Jacobian to a Host execution context.
   *
   * @param[in] ctx - Context providing Host vector operations.
   */
  explicit HostJacobian(Context<MemorySpace::Host>& ctx) noexcept;

  ~HostJacobian() override;

  HostJacobian(const HostJacobian&)            = delete;
  HostJacobian& operator=(const HostJacobian&) = delete;
  HostJacobian(HostJacobian&&)                 = delete;
  HostJacobian& operator=(HostJacobian&&)      = delete;

  void begin(const HostCsrPattern& pattern) override;
  void addElement(const ElementJacobianView& element) override;
  void replaceRows(HostVectorView<const Index> rows,
                   Real                        diagonal) override;
  void eliminateColumns(HostVectorView<const Index> rows,
                        HostVectorView<const Real>  values,
                        HostVectorView<Real>        rhs) override;
  void finalize() override;
  void apply(HostVectorView<const Real> direction,
             HostVector<Real>&          out) const override;
  void applyT(HostVectorView<const Real> direction,
              HostVector<Real>&          out) const override;

  /** @brief Return the owned CSR matrix for a native Host solver. */
  const HostCsrMatrix& matrix() const noexcept;

  /** @brief Construct a Host CSR transpose. */
  void transpose(const HostCsrMatrix& source,
                 HostCsrMatrix&       destination) const;

  /** @brief Apply an arbitrary Host CSR matrix. */
  void apply(const HostCsrMatrix&       matrix,
             HostVectorView<const Real> direction,
             HostVectorView<Real>       out,
             Real                       alpha = 1.0,
             Real                       beta  = 0.0) const;

  /** @brief Apply the transpose of an arbitrary Host CSR matrix. */
  void applyT(const HostCsrMatrix&       matrix,
              HostVectorView<const Real> direction,
              HostVectorView<Real>       out,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const;

  /** @brief Apply an arbitrary row-major Host dense matrix. */
  void apply(HostMatrixView<const Real> matrix,
             HostVectorView<const Real> direction,
             HostVectorView<Real>       out,
             Real                       alpha = 1.0,
             Real                       beta  = 0.0) const;

  /** @brief Apply the transpose of a Host dense matrix. */
  void applyT(HostMatrixView<const Real> matrix,
              HostVectorView<const Real> direction,
              HostVectorView<Real>       out,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const;

private:
  class ConstraintCache;

  ConstraintCache& constraints(HostVectorView<const Index> rows);

  Context<MemorySpace::Host>&      ctx_;
  HostCsrMatrix                    matrix_;
  std::unique_ptr<ConstraintCache> constraints_;
};

/// @cond INTERNAL
namespace detail
{

void applyHost(const HostCsrMatrix&       matrix,
               HostVectorView<const Real> direction,
               HostVectorView<Real>       out,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0);

void applyHostT(const HostCsrMatrix&       matrix,
                HostVectorView<const Real> direction,
                HostVectorView<Real>       out,
                Real                       alpha = 1.0,
                Real                       beta  = 0.0);

void applyHost(HostMatrixView<const Real> matrix,
               HostVectorView<const Real> direction,
               HostVectorView<Real>       out,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0);

void applyHostT(HostMatrixView<const Real> matrix,
                HostVectorView<const Real> direction,
                HostVectorView<Real>       out,
                Real                       alpha = 1.0,
                Real                       beta  = 0.0);

void transposeHostCsr(const HostCsrMatrix& source, HostCsrMatrix& destination);

} // namespace detail

/// @endcond

} // namespace femx::linalg
