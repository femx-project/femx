#include <cstddef>
#include <stdexcept>
#include <string>

#include <femx/assembly/Assembler.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx
{
namespace assembly
{

Assembler::Assembler(DofLayout space, AssemblyMode mode)
  : row_layout_(space),
    col_layout_(space),
    same_layout_(true),
    mode_(mode)
{
}

Assembler::Assembler(const FESpace& space, AssemblyMode mode)
  : Assembler(DofLayout(space), mode)
{
}

Assembler::Assembler(const MixedFESpace& space, AssemblyMode mode)
  : Assembler(DofLayout(space), mode)
{
}

Assembler::Assembler(DofLayout row_layout,
                     DofLayout col_layout,
                     AssemblyMode mode)
  : row_layout_(row_layout),
    col_layout_(col_layout),
    same_layout_(false),
    mode_(mode)
{
  checkCellCounts();
}

Assembler::Assembler(const FESpace& row_space,
                     const FESpace& col_space,
                     AssemblyMode mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Assembler::Assembler(const FESpace& row_space,
                     const MixedFESpace& col_space,
                     AssemblyMode mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Assembler::Assembler(const MixedFESpace& row_space,
                     const FESpace& col_space,
                     AssemblyMode mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Assembler::Assembler(const MixedFESpace& row_space,
                     const MixedFESpace& col_space,
                     AssemblyMode mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Index Assembler::numCells() const
{
  return row_layout_.numElems();
}

Index Assembler::numRows() const
{
  return row_layout_.numDofs();
}

Index Assembler::numCols() const
{
  return col_layout_.numDofs();
}

void Assembler::initVector(Vector<Real>& out) const
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

void Assembler::initMatrix(algebra::MatrixBuilder& out) const
{
  out.resize(numRows(), numCols());
  out.setZero();
}

void Assembler::addVector(Index ic,
                          const Vector<Real>& local,
                          Vector<Real>& out) const
{
  if (out.size() != numRows())
  {
    throw std::runtime_error("Assembler global vector has incompatible size");
  }

  Vector<Index> row_dofs;
  row_layout_.elemDofs(ic, row_dofs);
  if (local.size() != row_dofs.size())
  {
    throw std::runtime_error(
        "Assembler local vector size does not match row dofs");
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index row = row_dofs[i];
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

void Assembler::addMatrix(Index ic,
                          const DenseMatrix& local,
                          algebra::MatrixBuilder& out) const
{
  if (out.numRows() != numRows() || out.numCols() != numCols())
  {
    throw std::runtime_error("Assembler global matrix has incompatible size");
  }

  if (same_layout_ && out.addLocalMatrix(ic, local, mode_ == AssemblyMode::Atomic))
  {
    return;
  }

  Vector<Index> row_dofs;
  Vector<Index> col_dofs;
  row_layout_.elemDofs(ic, row_dofs);
  col_layout_.elemDofs(ic, col_dofs);

  if (local.rows() != row_dofs.size() || local.cols() != col_dofs.size())
  {
    throw std::runtime_error(
        "Assembler local matrix size does not match elem dofs");
  }

  for (Index i = 0; i < local.rows(); ++i)
  {
    const Index row = row_dofs[i];
    checkDof(row, numRows(), "row");
    for (Index j = 0; j < local.cols(); ++j)
    {
      const Index col = col_dofs[j];
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

void Assembler::checkCellCounts() const
{
  if (row_layout_.numElems() != col_layout_.numElems())
  {
    throw std::runtime_error(
        "Assembler row and column layouts have different cell counts");
  }
}

void Assembler::checkDof(Index dof, Index size, const char* name)
{
  if (dof < 0 || dof >= size)
  {
    throw std::runtime_error(
        std::string("Assembler ") + name + " dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
