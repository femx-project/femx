#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <refem/common/Types.hpp>

namespace refem
{

class Vector
{
public:
  Vector() = default;

  explicit Vector(index_type size)
    : values_(static_cast<std::size_t>(size), real_type{})
  {
  }

  void resize(index_type size)
  {
    values_.assign(static_cast<std::size_t>(size), real_type{});
  }

  index_type size() const
  {
    return static_cast<index_type>(values_.size());
  }

  real_type& operator[](index_type i)
  {
    return values_[static_cast<std::size_t>(i)];
  }

  real_type operator[](index_type i) const
  {
    return values_[static_cast<std::size_t>(i)];
  }

  real_type* data()
  {
    return values_.data();
  }

  const real_type* data() const
  {
    return values_.data();
  }

  void setZero()
  {
    std::fill(values_.begin(), values_.end(), real_type{});
  }

  void addLocalVector(const std::vector<index_type>& dofs,
                      const Vector&                  local)
  {
    if (static_cast<index_type>(dofs.size()) != local.size())
    {
      throw std::runtime_error("Local vector size does not match cell dofs");
    }

    for (index_type i = 0; i < local.size(); ++i)
    {
      const index_type dof = dofs[static_cast<std::size_t>(i)];

      if (dof < 0 || dof >= size())
      {
        throw std::runtime_error("Vector dof is out of range");
      }

      values_[static_cast<std::size_t>(dof)] += local[i];
    }
  }

private:
  std::vector<real_type> values_;
};

} // namespace refem
