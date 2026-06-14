#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixBackend.hpp>

namespace femx
{

class DenseMatrix;

class SparseMatrixImpl
{
public:
  virtual ~SparseMatrixImpl() = default;

  virtual void setZero() = 0;

  virtual Index rows() const = 0;
  virtual Index cols() const = 0;
  virtual Index nnz() const  = 0;

  virtual MatrixBackend backend() const = 0;

  virtual const Index* rowPtrData() const = 0;
  virtual const Index* colIndData() const = 0;
  virtual Real*        valuesData()       = 0;
  virtual const Real*  valuesData() const = 0;
};

} // namespace femx
