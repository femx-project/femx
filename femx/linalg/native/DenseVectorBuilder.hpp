#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace linalg
{

/** @brief In-memory mutable vector assembly target. */
class DenseVectorBuilder final
{
public:
  explicit DenseVectorBuilder(Index size = 0)
    : vec_(size)
  {
  }

  Index size() const
  {
    return vec_.size();
  }

  void resize(Index size)
  {
    vec_.resize(size);
  }

  void setZero()
  {
    vec_.setZero();
  }

  void set(Index row, Real value)
  {
    checkIndex(row);
    vec_[row] = value;
  }

  void add(Index row, Real value)
  {
    checkIndex(row);
    vec_[row] += value;
  }

  void addAtomic(Index row, Real value)
  {
    checkIndex(row);
    Real* vals = vec_.data();
#pragma omp atomic update
    vals[row] += value;
  }

  void finalize()
  {
  }

  Vector<Real>& vector()
  {
    return vec_;
  }

  const Vector<Real>& vector() const
  {
    return vec_;
  }

private:
  void checkIndex(Index row) const
  {
    if (row < 0 || row >= size())
    {
      throw std::runtime_error("DenseVectorBuilder index is out of range");
    }
  }

private:
  Vector<Real> vec_;
};

} // namespace linalg
} // namespace femx
