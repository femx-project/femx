#pragma once

#include <cstddef>
#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/system/SystemVector.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace system
{

/** @brief In-memory SystemVector backed by femx::Vector. */
class DenseSystemVector final : public SystemVector
{
public:
  explicit DenseSystemVector(index_type size = 0)
    : vector_(size)
  {
  }

  index_type size() const override
  {
    return vector_.size();
  }

  void resize(index_type size) override
  {
    vector_.resize(size);
  }

  void setZero() override
  {
    vector_.setZero();
  }

  void set(index_type row, real_type value) override
  {
    checkIndex(row);
    vector_[row] = value;
  }

  void add(index_type row, real_type value) override
  {
    checkIndex(row);
    vector_[row] += value;
  }

  void addAtomic(index_type row, real_type value) override
  {
    checkIndex(row);
    real_type* values = vector_.data();
#pragma omp atomic update
    values[static_cast<std::size_t>(row)] += value;
  }

  void finalize() override
  {
  }

  Vector& vector()
  {
    return vector_;
  }

  const Vector& vector() const
  {
    return vector_;
  }

private:
  void checkIndex(index_type row) const
  {
    if (row < 0 || row >= size())
    {
      throw std::runtime_error("DenseSystemVector index is out of range");
    }
  }

private:
  Vector vector_;
};

} // namespace system
} // namespace femx
