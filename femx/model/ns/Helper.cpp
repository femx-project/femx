#include "Helper.hpp"

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

namespace femx::model::ns
{

unique_ptr<FiniteElement> makeElement(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw runtime_error("Mesh has no elems");
  }

  const Element::Shape shape = mesh.elems().front().shape();
  if (shape == Element::Shape::Quadrilateral)
  {
    return make_unique<LagrangeQuadQ1>();
  }
  if (shape == Element::Shape::Triangle)
  {
    return make_unique<LagrangeTriangleP1>();
  }
  if (shape == Element::Shape::Tetrahedron)
  {
    return make_unique<LagrangeTetrahedronP1>();
  }
  throw runtime_error("Unsupported mesh elem type for Navier app");
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
  Index      in_out   = 0;
  Real       dist_out = 0.0;
  const auto cen      = selectorCenter(space.mesh(), sel);
  for (Index in = 0; in < space.mesh().numNodes(); ++in)
  {
    const Real dist = sqDist(space.mesh().node(in), cen);
    if (in == 0 || dist < dist_out)
    {
      in_out   = in;
      dist_out = dist;
    }
  }

  Vector<Index> dofs(1);
  dofs[0] = space.field(1).globalDof(in_out, 0);
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

} // namespace femx::model::ns
