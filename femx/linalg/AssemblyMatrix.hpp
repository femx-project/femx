#pragma once

#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/MatrixBuilder.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Matrix that can be assembled and applied as a linear operator.
 *
 * AssemblyMatrix combines the mutable MatrixBuilder interface with
 * LinearOperator application for assembled systems.
 */
class AssemblyMatrix : public LinearOperator, public MatrixBuilder
{
public:
  ~AssemblyMatrix() override = default;

  Index numRows() const override = 0;
  Index numCols() const override = 0;

  void matvec(const Vector<Real>& x, Vector<Real>& out) const
  {
    apply(x, out);
  }

  void matvecT(const Vector<Real>& x, Vector<Real>& out) const
  {
    applyT(x, out);
  }
};

} // namespace linalg
} // namespace femx
