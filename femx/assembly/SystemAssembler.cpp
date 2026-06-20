#include <stdexcept>
#include <string>

#include <femx/assembly/SystemAssembler.hpp>

using namespace femx::system;

namespace femx
{
namespace assembly
{

SystemAssembler::SystemAssembler(DofLayout    space,
                                 AssemblyMode mode)
  : row_layout_(space),
    col_layout_(space),
    same_layout_(true),
    mode_(mode)
{
}

SystemAssembler::SystemAssembler(const FESpace& space,
                                 AssemblyMode   mode)
  : SystemAssembler(DofLayout(space), mode)
{
}

SystemAssembler::SystemAssembler(const MixedFESpace& space,
                                 AssemblyMode        mode)
  : SystemAssembler(DofLayout(space), mode)
{
}

SystemAssembler::SystemAssembler(DofLayout    row_layout,
                                 DofLayout    col_layout,
                                 AssemblyMode mode)
  : row_layout_(row_layout),
    col_layout_(col_layout),
    same_layout_(false),
    mode_(mode)
{
  checkCellCounts();
}

SystemAssembler::SystemAssembler(const FESpace& row_space,
                                 const FESpace& col_space,
                                 AssemblyMode   mode)
  : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

SystemAssembler::SystemAssembler(const FESpace&      row_space,
                                 const MixedFESpace& col_space,
                                 AssemblyMode        mode)
  : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

SystemAssembler::SystemAssembler(const MixedFESpace& row_space,
                                 const FESpace&      col_space,
                                 AssemblyMode        mode)
  : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

SystemAssembler::SystemAssembler(const MixedFESpace& row_space,
                                 const MixedFESpace& col_space,
                                 AssemblyMode        mode)
  : SystemAssembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Index SystemAssembler::numCells() const
{
  return row_layout_.numElems();
}

Index SystemAssembler::numRows() const
{
  return row_layout_.numDofs();
}

Index SystemAssembler::numCols() const
{
  return col_layout_.numDofs();
}

void SystemAssembler::initVec(Vector<Real>& out) const
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

void SystemAssembler::initVec(SystemVector& out) const
{
  out.resize(numRows());
  out.setZero();
}

void SystemAssembler::initMat(SystemMatrix& out) const
{
  out.resize(numRows(), numCols());
  out.setZero();
}

void SystemAssembler::addVec(Index               ic,
                             const Vector<Real>& local,
                             Vector<Real>&       out) const
{
  if (out.size() != numRows())
  {
    throw std::runtime_error(
        "SystemAssembler global vector has incompatible size");
  }

  Vector<Index> row_dofs;
  row_layout_.elemDofs(ic, row_dofs);
  if (local.size() != row_dofs.size())
  {
    throw std::runtime_error(
        "SystemAssembler local vector size does not match row dofs");
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

void SystemAssembler::addVec(Index               ic,
                             const Vector<Real>& local,
                             SystemVector&       out) const
{
  if (out.size() != numRows())
  {
    throw std::runtime_error(
        "SystemAssembler global system vector has incompatible size");
  }

  Vector<Index> row_dofs;
  row_layout_.elemDofs(ic, row_dofs);
  if (local.size() != row_dofs.size())
  {
    throw std::runtime_error(
        "SystemAssembler local vector size does not match row dofs");
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index row = row_dofs[i];
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

void SystemAssembler::addMat(Index              ic,
                             const DenseMatrix& local,
                             SystemMatrix&      out) const
{
  if (out.numRows() != numRows() || out.numCols() != numCols())
  {
    throw std::runtime_error(
        "SystemAssembler global matrix has incompatible size");
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
        "SystemAssembler local matrix size does not match elem dofs");
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

void SystemAssembler::checkCellCounts() const
{
  if (row_layout_.numElems() != col_layout_.numElems())
  {
    throw std::runtime_error(
        "SystemAssembler row and column layouts have different cell counts");
  }
}

void SystemAssembler::checkDof(Index       dof,
                               Index       size,
                               const char* name)
{
  if (dof < 0 || dof >= size)
  {
    throw std::runtime_error(std::string("SystemAssembler ") + name
                             + " dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
