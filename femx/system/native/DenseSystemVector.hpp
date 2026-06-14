#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemVector.hpp>

namespace femx
{
namespace system
{

/** @brief In-memory SystemVector backed by femx::Vector. */
class DenseSystemVector final : public SystemVector
{
public:
  explicit DenseSystemVector(Index size = 0)
    : vec_(size)
  {
  }

  Index size() const override
  {
    return vec_.size();
  }

  void resize(Index size) override
  {
    vec_.resize(size);
  }

  void setZero() override
  {
    vec_.setZero();
  }

  void set(Index row, Real value) override
  {
    checkIndex(row);
    vec_[row] = value;
  }

  void add(Index row, Real value) override
  {
    checkIndex(row);
    vec_[row] += value;
  }

  void addAtomic(Index row, Real value) override
  {
    checkIndex(row);
    Real* values = vec_.data();
#pragma omp atomic update
    values[static_cast<std::size_t>(row)] += value;
  }

  void finalize() override
  {
  }

  Vector& vector()
  {
    return vec_;
  }

  const Vector& vector() const
  {
    return vec_;
  }

private:
  void checkIndex(Index row) const
  {
    if (row < 0 || row >= size())
    {
      throw std::runtime_error("DenseSystemVector index is out of range");
    }
  }

private:
  Vector vec_;
};

} // namespace system
} // namespace femx
