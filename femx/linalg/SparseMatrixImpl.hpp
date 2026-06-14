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

  virtual index_type rows() const = 0;
  virtual index_type cols() const = 0;
  virtual index_type nnz() const  = 0;

  virtual MatrixBackend backend() const = 0;

  virtual const index_type* rowPtrData() const = 0;
  virtual const index_type* colIndData() const = 0;
  virtual real_type*        valuesData()       = 0;
  virtual const real_type*  valuesData() const = 0;
};

} // namespace femx
