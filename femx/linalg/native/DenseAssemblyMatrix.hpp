#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/AssemblyMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Dense in-memory assembly matrix.
 *
 * DenseAssemblyMatrix stores assembled entries explicitly and provides
 * LinearOperator application for small systems.
 */
class DenseAssemblyMatrix final : public AssemblyMatrix
{
public:
  Index numRows() const override;
  Index numCols() const override;

  void resize(Index rows, Index cols) override;
  void setZero() override;
  void set(Index row, Index col, Real value) override;
  void add(Index row, Index col, Real value) override;
  void addAtomic(Index row, Index col, Real value) override;
  void finalize() override;

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override;
  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override;

  DenseMatrix&       mat();
  const DenseMatrix& mat() const;

private:
  DenseMatrix mat_;
};

} // namespace linalg
} // namespace femx
