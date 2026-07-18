#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class DenseMatrix final : public linalg::MatrixOperator
{
public:
  DenseMatrix();

  DenseMatrix(Index rows, Index cols);

  void resize(Index rows, Index cols) override;

  void setZero() override;

  Index numRows() const override;
  Index numCols() const override;

  void set(Index row, Index col, Real val) override;
  void add(Index row, Index col, Real val) override;
  void addAtomic(Index row, Index col, Real val) override;
  void finalize() override;

  Index size() const;

  Real& operator()(Index i, Index j);
  Real  operator()(Index i, Index j) const;

  void apply(const HostVector& x, HostVector& out) const override;
  void applyT(const HostVector& x, HostVector& out) const override;

  Real*       data();
  const Real* data() const;

private:
  Index rows_;
  Index cols_;

  HostVector vals_;
};

} // namespace femx
