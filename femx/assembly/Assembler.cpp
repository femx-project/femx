#include <cstddef>
#include <stdexcept>
#include <string>

#include <femx/assembly/Assembler.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>

using namespace std;
using namespace femx::linalg;

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

Assembler::Assembler(DofLayout    row_layout,
                     DofLayout    col_layout,
                     AssemblyMode mode)
  : row_layout_(row_layout),
    col_layout_(col_layout),
    same_layout_(false),
    mode_(mode)
{
  checkElemCounts();
}

Assembler::Assembler(const FESpace& row_space,
                     const FESpace& col_space,
                     AssemblyMode   mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Assembler::Assembler(const FESpace&      row_space,
                     const MixedFESpace& col_space,
                     AssemblyMode        mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Assembler::Assembler(const MixedFESpace& row_space,
                     const FESpace&      col_space,
                     AssemblyMode        mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Assembler::Assembler(const MixedFESpace& row_space,
                     const MixedFESpace& col_space,
                     AssemblyMode        mode)
  : Assembler(DofLayout(row_space), DofLayout(col_space), mode)
{
}

Index Assembler::numElems() const
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

void Assembler::initVec(Vector<Real>& out) const
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

void Assembler::initMat(MatrixBuilder& out) const
{
  out.resize(numRows(), numCols());
  out.setZero();
}

void Assembler::addVec(Index               ie,
                       const Vector<Real>& local,
                       Vector<Real>&       out) const
{
  if (out.size() != numRows())
  {
    throw runtime_error("Assembler global vector has incompatible size");
  }

  Vector<Index> row_dofs;
  row_layout_.elemDofs(ie, row_dofs);
  if (local.size() != row_dofs.size())
  {
    throw runtime_error(
        "Assembler local vector size does not match row dofs");
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index row = row_dofs[i];
    checkDof(row, numRows(), "row");
    if (mode_ == AssemblyMode::Atomic)
    {
      Real* vals = out.data();
#pragma omp atomic update
      vals[row] += local[i];
    }
    else
    {
      out[row] += local[i];
    }
  }
}

void Assembler::addMat(Index              ie,
                       const DenseMatrix& local,
                       MatrixBuilder&     out) const
{
  if (out.numRows() != numRows() || out.numCols() != numCols())
  {
    throw runtime_error("Assembler global matrix has incompatible size");
  }

  const bool atomic = mode_ == AssemblyMode::Atomic;
  if (same_layout_ && out.addMappedMat(ie, local, atomic))
  {
    return;
  }

  Vector<Index> row_dofs;
  Vector<Index> col_dofs;
  row_layout_.elemDofs(ie, row_dofs);
  col_layout_.elemDofs(ie, col_dofs);

  if (local.rows() != row_dofs.size() || local.cols() != col_dofs.size())
  {
    throw runtime_error("Assembler local matrix size does not match elem dofs");
  }

  out.addMat(row_dofs, col_dofs, local, atomic);
}

void Assembler::checkElemCounts() const
{
  if (row_layout_.numElems() != col_layout_.numElems())
  {
    throw runtime_error(
        "Assembler row and column layouts have different elem counts");
  }
}

void Assembler::checkDof(Index id, Index size, const char* name)
{
  if (id < 0 || id >= size)
  {
    throw runtime_error(
        string("Assembler ") + name + " id is out of range");
  }
}

} // namespace assembly
} // namespace femx
