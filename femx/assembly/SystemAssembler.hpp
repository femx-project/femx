#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <femx/assembly/DofLayout.hpp>
#include <femx/core/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/system/SystemVector.hpp>

namespace femx
{
namespace assembly
{

/** @brief Scatter-adds elem vectors and matrices into equation objects. */
class SystemAssembler
{
public:
  enum class AssemblyMode
  {
    Serial,
    Atomic
  };

  explicit SystemAssembler(DofLayout    space,
                           AssemblyMode mode = AssemblyMode::Serial)
    : row_layout_(space),
      col_layout_(space),
      mode_(mode)
  {
  }

  explicit SystemAssembler(const FESpace& space,
                           AssemblyMode   mode = AssemblyMode::Serial)
    : SystemAssembler(DofLayout(space), mode)
  {
  }

  explicit SystemAssembler(const MixedFESpace& space,
                           AssemblyMode        mode = AssemblyMode::Serial)
    : SystemAssembler(DofLayout(space), mode)
  {
  }

  SystemAssembler(DofLayout    row_layout,
                  DofLayout    col_layout,
                  AssemblyMode mode = AssemblyMode::Serial)
    : row_layout_(row_layout),
      col_layout_(col_layout),
      mode_(mode)
  {
    checkCellCounts();
  }

  SystemAssembler(const FESpace& row_space,
                  const FESpace& col_space,
                  AssemblyMode   mode = AssemblyMode::Serial)
    : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
  {
  }

  SystemAssembler(const FESpace&      row_space,
                  const MixedFESpace& col_space,
                  AssemblyMode        mode = AssemblyMode::Serial)
    : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
  {
  }

  SystemAssembler(const MixedFESpace& row_space,
                  const FESpace&      col_space,
                  AssemblyMode        mode = AssemblyMode::Serial)
    : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
  {
  }

  SystemAssembler(const MixedFESpace& row_space,
                  const MixedFESpace& col_space,
                  AssemblyMode        mode = AssemblyMode::Serial)
    : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
  {
  }

  index_type numCells() const
  {
    return row_layout_.numElems();
  }

  index_type numRows() const
  {
    return row_layout_.numDofs();
  }

  index_type numCols() const
  {
    return col_layout_.numDofs();
  }

  void initVec(Vector& out) const
  {
    if (out.size() != numRows())
    {
      out.resize(numRows());
    }
    else
    {
      out.setZero();
    }
  }

  void initVec(system::SystemVector& out) const
  {
    out.resize(numRows());
    out.setZero();
  }

  void initMat(system::SystemMatrix& out) const
  {
    out.resize(numRows(), numCols());
    out.setZero();
  }

  void addVec(index_type cell, const Vector& local, Vector& out) const
  {
    if (out.size() != numRows())
    {
      throw std::runtime_error(
          "SystemAssembler global vector has incompatible size");
    }

    std::vector<index_type> row_dofs;
    row_layout_.elemDofs(cell, row_dofs);
    if (local.size() != static_cast<index_type>(row_dofs.size()))
    {
      throw std::runtime_error(
          "SystemAssembler local vector size does not match row dofs");
    }

    for (index_type i = 0; i < local.size(); ++i)
    {
      const index_type row = row_dofs[static_cast<std::size_t>(i)];
      checkDof(row, numRows(), "row");
      if (mode_ == AssemblyMode::Atomic)
      {
        real_type* values = out.data();
#pragma omp atomic update
        values[static_cast<std::size_t>(row)] += local[i];
      }
      else
      {
        out[row] += local[i];
      }
    }
  }

  void addVec(index_type            cell,
              const Vector&         local,
              system::SystemVector& out) const
  {
    if (out.size() != numRows())
    {
      throw std::runtime_error(
          "SystemAssembler global system vector has incompatible size");
    }

    std::vector<index_type> row_dofs;
    row_layout_.elemDofs(cell, row_dofs);
    if (local.size() != static_cast<index_type>(row_dofs.size()))
    {
      throw std::runtime_error(
          "SystemAssembler local vector size does not match row dofs");
    }

    for (index_type i = 0; i < local.size(); ++i)
    {
      const index_type row = row_dofs[static_cast<std::size_t>(i)];
      checkDof(row, numRows(), "row");
      if (mode_ == AssemblyMode::Atomic)
      {
        out.addAtomic(row, local[i]);
      }
      else
      {
        out.add(row, local[i]);
      }
    }
  }

  void addMat(index_type            cell,
              const DenseMatrix&    local,
              system::SystemMatrix& out) const
  {
    if (out.numRows() != numRows() || out.numCols() != numCols())
    {
      throw std::runtime_error(
          "SystemAssembler global matrix has incompatible size");
    }

    std::vector<index_type> row_dofs;
    std::vector<index_type> col_dofs;
    row_layout_.elemDofs(cell, row_dofs);
    col_layout_.elemDofs(cell, col_dofs);

    if (local.rows() != static_cast<index_type>(row_dofs.size())
        || local.cols() != static_cast<index_type>(col_dofs.size()))
    {
      throw std::runtime_error(
          "SystemAssembler local matrix size does not match elem dofs");
    }

    for (index_type i = 0; i < local.rows(); ++i)
    {
      const index_type row = row_dofs[static_cast<std::size_t>(i)];
      checkDof(row, numRows(), "row");
      for (index_type j = 0; j < local.cols(); ++j)
      {
        const index_type col = col_dofs[static_cast<std::size_t>(j)];
        checkDof(col, numCols(), "column");
        if (mode_ == AssemblyMode::Atomic)
        {
          out.addAtomic(row, col, local(i, j));
        }
        else
        {
          out.add(row, col, local(i, j));
        }
      }
    }
  }

private:
  void checkCellCounts() const
  {
    if (row_layout_.numElems() != col_layout_.numElems())
    {
      throw std::runtime_error(
          "SystemAssembler row and column layouts have different cell counts");
    }
  }

  static void checkDof(index_type dof, index_type size, const char* name)
  {
    if (dof < 0 || dof >= size)
    {
      throw std::runtime_error(std::string("SystemAssembler ") + name
                               + " dof is out of range");
    }
  }

private:
  DofLayout    row_layout_;
  DofLayout    col_layout_;
  AssemblyMode mode_{AssemblyMode::Serial};
};

} // namespace assembly
} // namespace femx
