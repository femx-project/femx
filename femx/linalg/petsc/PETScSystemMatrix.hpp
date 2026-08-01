#pragma once

#include <femx/linalg/SystemMatrix.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>
#include <femx/linalg/petsc/PETScMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own and assemble a PETSc-native Host system matrix.
 */
class PETScSystemMatrix final : public SystemMatrix<MemorySpace::Host>
{
  using Base = SystemMatrix<MemorySpace::Host>;

public:
  /**
   * @brief Construct a PETSc system matrix on an MPI context communicator.
   *
   * @param[in] ctx - System-owned MPI execution context.
   */
  explicit PETScSystemMatrix(MpiContext& ctx);

  /**
   * @copydoc Base::setup()
   */
  void setup(const HostCsrPattern& pattern) override;

  /**
   * @copydoc Base::addElement()
   */
  void addElement(const ElementJacobianView& element) override;

  /**
   * @copydoc Base::replaceRows()
   */
  void replaceRows(HostVectorView<const Index> rows,
                   Real                        diag) override;

  /**
   * @copydoc Base::eliminateColumns()
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
   * @brief Return the owned PETSc matrix for the native solver.
   */
  const PETScMatrix& matrix() const noexcept;

private:
  MpiContext& ctx_;
  PETScMatrix mat_;
};

} // namespace femx::linalg
