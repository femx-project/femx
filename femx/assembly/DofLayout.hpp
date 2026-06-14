#pragma once

#include <stdexcept>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx
{
namespace assembly
{

/** @brief Non-owning view of elem-to-global dof connectivity. */
class DofLayout
{
public:
  explicit DofLayout(const FESpace& space)
    : fe_space_(&space)
  {
  }

  explicit DofLayout(const MixedFESpace& space)
    : mixed_space_(&space)
  {
  }

  Index numElems() const
  {
    if (fe_space_ != nullptr)
    {
      return fe_space_->numElems();
    }
    return mixedSpace().numElems();
  }

  Index numDofs() const
  {
    if (fe_space_ != nullptr)
    {
      return fe_space_->numDofs();
    }
    return mixedSpace().numDofs();
  }

  Index numDofsPerElem() const
  {
    if (fe_space_ != nullptr)
    {
      return fe_space_->numDofsPerElem();
    }
    return mixedSpace().numDofsPerElem();
  }

  void elemDofs(Index ic, std::vector<Index>& dofs) const
  {
    if (ic < 0 || ic >= numElems())
    {
      throw std::runtime_error("DofLayout cell index is out of range");
    }
    if (fe_space_ != nullptr)
    {
      fe_space_->elemDofs(ic, dofs);
      return;
    }
    mixedSpace().elemDofs(ic, dofs);
  }

private:
  const MixedFESpace& mixedSpace() const
  {
    if (mixed_space_ == nullptr)
    {
      throw std::runtime_error("DofLayout is not initialized");
    }
    return *mixed_space_;
  }

private:
  const FESpace*      fe_space_{nullptr};
  const MixedFESpace* mixed_space_{nullptr};
};

} // namespace assembly
} // namespace femx
