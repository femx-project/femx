#pragma once

#include <refem/common/Types.hpp>
#include <refem/linalg/MatrixBackend.hpp>

namespace refem
{

class DenseMatrix;

class SparseMatrixImpl
{
public:
  virtual ~SparseMatrixImpl() = default;

  virtual void setZero() = 0;

  virtual void addLocalMatrix(index_type ic, const DenseMatrix& Ke) = 0;

  virtual index_type rows() const = 0;
  virtual index_type cols() const = 0;
  virtual index_type nnz() const  = 0;

  virtual MatrixBackend backend() const = 0;

  virtual const index_type* rowPtrData() const = 0;
  virtual const index_type* colIndData() const = 0;
  virtual real_type*        valuesData()       = 0;
  virtual const real_type*  valuesData() const = 0;
};

} // namespace refem
