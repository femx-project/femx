#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/AssemblyMatrix.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/**
 * @brief Sparse assembly matrix backed by native CSR storage.
 *
 * CsrAssemblyMatrix accumulates entries into a fixed sparsity pattern and
 * applies the resulting matrix in forward or transpose mode.
 */
class CsrAssemblyMatrix final : public AssemblyMatrix
{
public:
  explicit CsrAssemblyMatrix(const CsrPattern& pattern);

  Index numRows() const override;
  Index numCols() const override;

  void resize(Index rows, Index cols) override;
  void setZero() override;
  void set(Index row, Index col, Real value) override;
  void add(Index row, Index col, Real value) override;
  void addAtomic(Index row, Index col, Real value) override;
  bool addMappedMat(Index              ie,
                    const DenseMatrix& local,
                    bool               atomic) override;
  void finalize() override;

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override;
  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override;

  CsrMatrix&       mat();
  const CsrMatrix& mat() const;

private:
  void  checkMappedMat(Index ie, const DenseMatrix& local) const;
  void  addMappedMatSerial(Index ie, const DenseMatrix& local);
  void  addMappedMatAtomic(Index ie, const DenseMatrix& local);
  Index findEntry(Index row, Index col) const;

  CsrMatrix mat_;
};

} // namespace linalg
} // namespace femx
