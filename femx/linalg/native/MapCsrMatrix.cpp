#include <stdexcept>

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>

namespace femx
{
namespace linalg
{

MapCsrMatrix::MapCsrMatrix(
    const assembly::HostAssemblyMap& map)
  : mat_(map.graph()), map_(map)
{
}

Index MapCsrMatrix::numRows() const
{
  return mat_.rows();
}

Index MapCsrMatrix::numCols() const
{
  return mat_.cols();
}

void MapCsrMatrix::resize(Index rows, Index cols)
{
  if (rows != numRows() || cols != numCols())
  {
    throw std::runtime_error(
        "MapCsrMatrix cannot resize away from its AssemblyMap graph");
  }
}

void MapCsrMatrix::setZero()
{
  mat_.setZero();
}

void MapCsrMatrix::set(Index row, Index col, Real val)
{
  mat_.valsData()[findEntry(row, col)] = val;
}

void MapCsrMatrix::add(Index row, Index col, Real val)
{
  mat_.valsData()[findEntry(row, col)] += val;
}

void MapCsrMatrix::addAtomic(Index row, Index col, Real val)
{
  Real*       vals = mat_.valsData();
  const Index k    = findEntry(row, col);
#pragma omp atomic update
  vals[k] += val;
}

void MapCsrMatrix::finalize()
{
}

void MapCsrMatrix::apply(const HostVector& dir,
                         HostVector&       out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "MapCsrMatrix apply received incompatible vector");
  }

  resizeOrZero(out, numRows());
  const Index* rp   = mat_.rowPtrData();
  const Index* ci   = mat_.colIndData();
  const Real*  vals = mat_.valsData();

  for (Index row = 0; row < numRows(); ++row)
  {
    Real sum = 0.0;
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      sum += vals[k] * dir[ci[k]];
    }
    out[row] = sum;
  }
}

void MapCsrMatrix::applyT(const HostVector& dir,
                          HostVector&       out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "MapCsrMatrix transpose apply received incompatible vector");
  }

  resizeOrZero(out, numCols());
  const Index* rp   = mat_.rowPtrData();
  const Index* ci   = mat_.colIndData();
  const Real*  vals = mat_.valsData();

  for (Index row = 0; row < numRows(); ++row)
  {
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      out[ci[k]] += vals[k] * dir[row];
    }
  }
}

HostCsrMatrix& MapCsrMatrix::mat()
{
  return mat_;
}

const HostCsrMatrix& MapCsrMatrix::mat() const
{
  return mat_;
}

void MapCsrMatrix::checkElem(Index              ie,
                             const DenseMatrix& mat_e) const
{
  if (ie < 0 || ie >= map_.numElems())
  {
    throw std::runtime_error(
        "MapCsrMatrix element index is out of range");
  }

  const auto map = map_.view();
  if (mat_e.numRows() != map.numResDofs(ie)
      || mat_e.numCols() != map.numStateDofs(ie))
  {
    throw std::runtime_error(
        "MapCsrMatrix element matrix size does not match AssemblyMap");
  }
}

void MapCsrMatrix::addElem(Index ie,
                           const Array<Index>&,
                           const Array<Index>&,
                           const DenseMatrix& mat_e,
                           bool               atomic)
{
  checkElem(ie, mat_e);

  const auto  map    = map_.view();
  const Index first  = map.jac_offsets[ie];
  const Index n      = map.jac_offsets[ie + 1] - first;
  const Real* vals_e = mat_e.data();
  Real*       vals   = mat_.valsData();
  for (Index i = 0; i < n; ++i)
  {
    const Index k = map.jac_map[first + i];
    if (atomic)
    {
#pragma omp atomic update
      vals[k] += vals_e[i];
    }
    else
    {
      vals[k] += vals_e[i];
    }
  }
}

Index MapCsrMatrix::findEntry(Index row, Index col) const
{
  if (row < 0 || row >= numRows() || col < 0 || col >= numCols())
  {
    throw std::runtime_error("MapCsrMatrix index is out of range");
  }

  const Index* rp = mat_.rowPtrData();
  const Index* ci = mat_.colIndData();
  for (Index k = rp[row]; k < rp[row + 1]; ++k)
  {
    if (ci[k] == col)
    {
      return k;
    }
  }

  throw std::runtime_error(
      "MapCsrMatrix entry is outside its AssemblyMap graph");
}

} // namespace linalg
} // namespace femx
