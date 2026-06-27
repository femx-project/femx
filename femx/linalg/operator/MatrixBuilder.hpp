#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

namespace linalg
{

/** @brief Mutable assembly target for matrix entries. */
class MatrixBuilder
{
public:
  virtual ~MatrixBuilder() = default;

  virtual Index numRows() const = 0;
  virtual Index numCols() const = 0;

  virtual void resize(Index rows, Index cols)        = 0;
  virtual void setZero()                             = 0;
  virtual void set(Index row, Index col, Real value) = 0;
  virtual void add(Index row, Index col, Real value) = 0;
  virtual void finalize()                            = 0;

  virtual void addAtomic(Index row, Index col, Real value)
  {
    add(row, col, value);
  }

  virtual void addMat(const Vector<Index>& row_dofs,
                      const Vector<Index>& col_dofs,
                      const DenseMatrix&   local,
                      bool                 atomic)
  {
    if (local.rows() != row_dofs.size() || local.cols() != col_dofs.size())
    {
      throw std::runtime_error(
          "MatrixBuilder element matrix size does not match dofs");
    }

    const Index rows = numRows();
    const Index cols = numCols();

    for (Index i = 0; i < local.rows(); ++i)
    {
      const Index row = row_dofs[i];
      if (row < 0 || row >= rows)
      {
        throw std::runtime_error("MatrixBuilder row dof is out of range");
      }

      for (Index j = 0; j < local.cols(); ++j)
      {
        const Index col = col_dofs[j];
        if (col < 0 || col >= cols)
        {
          throw std::runtime_error(
              "MatrixBuilder column dof is out of range");
        }

        if (atomic)
        {
          addAtomic(row, col, local(i, j));
        }
        else
        {
          add(row, col, local(i, j));
        }
      }
    }
  }

  virtual bool addMappedMat(Index, const DenseMatrix&, bool)
  {
    return false;
  }
};

} // namespace linalg
} // namespace femx
