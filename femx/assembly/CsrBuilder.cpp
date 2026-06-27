#include <femx/assembly/CsrBuilder.hpp>

using namespace std;

namespace femx
{
namespace assembly
{

CsrPattern CsrBuilder::build(const FESpace& space)
{
  return build(DofLayout(space));
}

CsrPattern CsrBuilder::build(const MixedFESpace& space)
{
  return build(DofLayout(space));
}

CsrPattern CsrBuilder::build(DofLayout layout)
{
  return CsrPattern(layout.numDofs(),
                    layout.numDofs(),
                    layout.numElems(),
                    [&layout](Index ie, Vector<Index>& dofs)
                    {
                      layout.elemDofs(ie, dofs);
                    });
}

} // namespace assembly
} // namespace femx
