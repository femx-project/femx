#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <femx/assembly/DofLayout.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/system/SystemVector.hpp>

namespace femx
{
namespace assembly
{

enum class AssemblyMode
{
  Serial,
  Atomic
};

/** @brief Scatter-adds elem vectors and matrices into equation objects. */
class SystemAssembler
{
public:
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

  Index numCells() const
  {
    return row_layout_.numElems();
  }

  Index numRows() const
  {
    return row_layout_.numDofs();
  }

  Index numCols() const
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

  void addVec(Index ic, const Vector& local, Vector& out) const
  {
    if (out.size() != numRows())
    {
      throw std::runtime_error(
          "SystemAssembler global vector has incompatible size");
    }

    std::vector<Index> row_dofs;
    row_layout_.elemDofs(ic, row_dofs);
    if (local.size() != static_cast<Index>(row_dofs.size()))
    {
      throw std::runtime_error(
          "SystemAssembler local vector size does not match row dofs");
    }

    for (Index i = 0; i < local.size(); ++i)
    {
      const Index row = row_dofs[static_cast<std::size_t>(i)];
      checkDof(row, numRows(), "row");
      if (mode_ == AssemblyMode::Atomic)
      {
        Real* values = out.data();
#pragma omp atomic update
        values[static_cast<std::size_t>(row)] += local[i];
      }
      else
      {
        out[row] += local[i];
      }
    }
  }

  void addVec(Index ic, const Vector& local, system::SystemVector& out) const
  {
    if (out.size() != numRows())
    {
      throw std::runtime_error(
          "SystemAssembler global system vector has incompatible size");
    }

    std::vector<Index> row_dofs;
    row_layout_.elemDofs(ic, row_dofs);
    if (local.size() != static_cast<Index>(row_dofs.size()))
    {
      throw std::runtime_error(
          "SystemAssembler local vector size does not match row dofs");
    }

    for (Index i = 0; i < local.size(); ++i)
    {
      const Index row = row_dofs[static_cast<std::size_t>(i)];
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

  void addMat(Index ic, const DenseMatrix& local, system::SystemMatrix& out) const
  {
    if (out.numRows() != numRows() || out.numCols() != numCols())
    {
      throw std::runtime_error(
          "SystemAssembler global matrix has incompatible size");
    }

    std::vector<Index> row_dofs;
    std::vector<Index> col_dofs;
    row_layout_.elemDofs(ic, row_dofs);
    col_layout_.elemDofs(ic, col_dofs);

    if (local.rows() != static_cast<Index>(row_dofs.size())
        || local.cols() != static_cast<Index>(col_dofs.size()))
    {
      throw std::runtime_error(
          "SystemAssembler local matrix size does not match elem dofs");
    }

    for (Index i = 0; i < local.rows(); ++i)
    {
      const Index row = row_dofs[static_cast<std::size_t>(i)];
      checkDof(row, numRows(), "row");
      for (Index j = 0; j < local.cols(); ++j)
      {
        const Index col = col_dofs[static_cast<std::size_t>(j)];
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

  static void checkDof(Index dof, Index size, const char* name)
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
