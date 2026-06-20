#include <stdexcept>

#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx
{

DofLayout::DofLayout(const FESpace& space)
  : fe_space_(&space)
{
}

DofLayout::DofLayout(const MixedFESpace& space)
  : mixed_space_(&space)
{
}

Index DofLayout::numElems() const
{
  if (fe_space_ != nullptr)
  {
    return fe_space_->numElems();
  }
  return mixedSpace().numElems();
}

Index DofLayout::numDofs() const
{
  if (fe_space_ != nullptr)
  {
    return fe_space_->numDofs();
  }
  return mixedSpace().numDofs();
}

Index DofLayout::numDofsPerElem() const
{
  if (fe_space_ != nullptr)
  {
    return fe_space_->numDofsPerElem();
  }
  return mixedSpace().numDofsPerElem();
}

void DofLayout::elemDofs(Index ic, Vector<Index>& dofs) const
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

const MixedFESpace& DofLayout::mixedSpace() const
{
  if (mixed_space_ == nullptr)
  {
    throw std::runtime_error("DofLayout is not initialized");
  }
  return *mixed_space_;
}

} // namespace femx
