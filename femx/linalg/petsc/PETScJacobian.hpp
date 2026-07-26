#pragma once

#include <femx/linalg/Jacobian.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>
#include <femx/linalg/petsc/PETScMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own and assemble a PETSc-native Host Jacobian.
 */
class PETScJacobian final : public Jacobian<MemorySpace::Host>
{
public:
  /**
   * @brief Construct a PETSc Jacobian on an MPI context communicator.
   *
   * @param[in] ctx - System-owned MPI execution context.
   */
  explicit PETScJacobian(MpiContext& ctx);

  void setup(const HostCsrPattern& pattern) override;
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

  /** @brief Return the owned PETSc matrix for the native solver. */
  const PETScMatrix& matrix() const noexcept;

private:
  MpiContext& ctx_;
  PETScMatrix matrix_;
};

} // namespace femx::linalg
