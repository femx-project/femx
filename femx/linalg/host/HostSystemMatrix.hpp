#pragma once

#include <memory>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/SystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own and operate on a Host CSR system matrix.
 *
 * Constraint metadata derived from the current pattern and constrained rows
 * is cached by this object.
 */
class HostSystemMatrix final : public SystemMatrix<MemorySpace::Host>
{
  using Base = SystemMatrix<MemorySpace::Host>;

public:
  /**
   * @brief Bind the system matrix to a Host execution context.
   *
   * @param[in] ctx - Context providing Host vector operations.
   */
  explicit HostSystemMatrix(Context<MemorySpace::Host>& ctx) noexcept;

  ~HostSystemMatrix() override;

  HostSystemMatrix(const HostSystemMatrix&)            = delete;
  HostSystemMatrix& operator=(const HostSystemMatrix&) = delete;
  HostSystemMatrix(HostSystemMatrix&&)                 = delete;
  HostSystemMatrix& operator=(HostSystemMatrix&&)      = delete;

  /**
   * @copydoc Base::setup()
   */
  void setup(const HostCsrPattern& pattern) override;

  /**
   * @copydoc Base::addElement()
   *
   * @throws std::runtime_error If validation fails.
   */
  void addElement(const ElementJacobianView& elem) override;

  /**
   * @copydoc Base::replaceRows()
   *
   * @throws std::runtime_error If validation fails.
   */
  void replaceRows(HostVectorView<const Index> rows,
                   Real                        diag) override;

  /**
   * @copydoc Base::eliminateColumns()
   *
   * @throws std::runtime_error If validation fails.
   */
  void eliminateColumns(HostVectorView<const Index> rows,
                        HostVectorView<const Real>  values,
                        HostVectorView<Real>        rhs) override;

  /**
   * @copydoc Base::finalize()
   */
  void finalize() override;

  /**
   * @copydoc Base::matvec()
   */
  void matvec(HostVectorView<const Real> dir,
              HostVector<Real>&          out) const override;

  /**
   * @copydoc Base::matvecT()
   */
  void matvecT(HostVectorView<const Real> dir,
               HostVector<Real>&          out) const override;

  /**
   * @brief Return the owned CSR matrix for a native Host solver.
   */
  const HostCsrMatrix& matrix() const noexcept;

private:
  class ConstraintCache;

  ConstraintCache& constraints(HostVectorView<const Index> rows);

  Context<MemorySpace::Host>&      ctx_;         ///< Bound Host execution context.
  HostCsrMatrix                    mat_;         ///< Owned Host CSR matrix.
  std::unique_ptr<ConstraintCache> constraints_; ///< Cached constraint metadata.
};

} // namespace femx::linalg
