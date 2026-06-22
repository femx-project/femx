#include <stdexcept>

#include <femx/fem/DofMap.hpp>

using namespace std;

namespace femx
{

DofMap::DofMap(Index ne, Index ndpe)
{
  allocate(ne, ndpe);
}

void DofMap::allocate(Index ne, Index ndpe)
{
  if (ne < 0 || ndpe <= 0)
  {
    throw runtime_error("DofMap: invalid size");
  }

  ne_   = ne;
  ndpe_ = ndpe;
  elem_dofs_.resize(ne_ * ndpe_);
}

Index DofMap::numElements() const noexcept
{
  return ne_;
}

Index DofMap::numElementDofs() const noexcept
{
  return ndpe_;
}

Index DofMap::elementDof(Index ie, Index il) const noexcept
{
  return elem_dofs_[offset(ie, il)];
}

void DofMap::setElementDof(Index ie, Index il, Index gdof) noexcept
{
  elem_dofs_[offset(ie, il)] = gdof;
}

const Index* DofMap::elementDofsData(Index ie) const noexcept
{
  return elem_dofs_.data() + ie * ndpe_;
}

Index DofMap::offset(Index ie, Index il) const noexcept
{
  return ie * ndpe_ + il;
}

} // namespace femx
