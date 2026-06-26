#include <femx/assembly/SparsityPatternBuilder.hpp>

using namespace std;

namespace femx
{
namespace assembly
{

CsrPattern SparsityPatternBuilder::build(const FESpace& space)
{
  return CsrPattern(space.numDofs(), space.numDofs(), collect(space));
}

CsrPattern SparsityPatternBuilder::build(const MixedFESpace& space)
{
  return CsrPattern(space.numDofs(), space.numDofs(), collect(space));
}

IndexSetList SparsityPatternBuilder::collect(const FESpace& space)
{
  IndexSetList cdofs;
  cdofs.reserveSets(space.numElems());
  cdofs.reserveValues(space.numElems() * space.numDofsPerElem());
  for (Index ie = 0; ie < space.numElems(); ++ie)
  {
    cdofs.pushBack(space.elemDofs(ie));
  }
  return cdofs;
}

IndexSetList SparsityPatternBuilder::collect(const MixedFESpace& space)
{
  IndexSetList cdofs;
  cdofs.reserveSets(space.numElems());
  cdofs.reserveValues(space.numElems() * space.numDofsPerElem());
  for (Index ie = 0; ie < space.numElems(); ++ie)
  {
    cdofs.pushBack(space.elemDofs(ie));
  }
  return cdofs;
}

} // namespace assembly
} // namespace femx
