#include "NavierHelper.hpp"

#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/VelocityProfile.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>

using namespace std;
using namespace femx;
using namespace femx::fem;

namespace femx::navier
{

unique_ptr<FiniteElement> makeElement(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw runtime_error("Mesh has no cells");
  }

  const Cell::Shape shape = mesh.cells().front().shape();
  if (shape == Cell::Shape::Quadrilateral)
  {
    return make_unique<LagrangeQuadQ1>();
  }
  if (shape == Cell::Shape::Triangle)
  {
    return make_unique<LagrangeTriangleP1>();
  }
  if (shape == Cell::Shape::Tetrahedron)
  {
    return make_unique<LagrangeTetrahedronP1>();
  }
  throw runtime_error("Unsupported mesh cell type for Navier app");
}

MixedFESpace makeSpace(Mesh& mesh, FiniteElement& elem)
{
  FESpace u_space(&mesh, &elem, mesh.dim());
  FESpace p_space(&mesh, &elem);

  MixedFESpace space;
  space.addField(u_space);
  space.addField(p_space);
  space.setup();
  return space;
}

Point3 selectorCenter(const Mesh& mesh, const BoundarySelector& sel)
{
  if (!sel.name.empty())
  {
    return boundaryCenter(mesh, sel.name);
  }
  return boundaryCenter(mesh, sel.physical);
}

Vector<Index> gaugeDofs(const MixedFESpace&     space,
                        const BoundarySelector& sel)
{
  Index      node_out = 0;
  Real       dist_out = 0.0;
  const auto cen      = selectorCenter(space.mesh(), sel);
  for (Index node = 0; node < space.mesh().numNodes(); ++node)
  {
    const Real dist = sqDist(space.mesh().node(node), cen);
    if (node == 0 || dist < dist_out)
    {
      node_out = node;
      dist_out = dist;
    }
  }

  Vector<Index> dofs(1);
  dofs[0] = space.field(1).globalDof(node_out, 0);
  return dofs;
}

DirichletControl makeVelocityControl(
    const MixedFESpace&     space,
    const BoundarySelector& sel)
{
  if (!sel.name.empty())
  {
    return makeVelocityControl(space, sel.name);
  }
  return makeVelocityControl(space, sel.physical);
}

} // namespace femx::navier
