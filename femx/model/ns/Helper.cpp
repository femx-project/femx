#include "Helper.hpp"

#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/VelocityProfile.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>
using namespace femx;
using namespace femx::fem;

namespace femx::model::ns
{

std::unique_ptr<fem::FiniteElement> makeElement(const fem::Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("Mesh has no elems");
  }

  const fem::Element::Shape shape = mesh.elems().front().shape();
  if (shape == fem::Element::Shape::Quadrilateral)
  {
    return std::make_unique<fem::LagrangeQuadQ1>();
  }
  if (shape == fem::Element::Shape::Triangle)
  {
    return std::make_unique<fem::LagrangeTriangleP1>();
  }
  if (shape == fem::Element::Shape::Tetrahedron)
  {
    return std::make_unique<fem::LagrangeTetrahedronP1>();
  }
  throw std::runtime_error("Unsupported mesh elem type for Navier app");
}

fem::MixedFESpace makeSpace(fem::Mesh& mesh, fem::FiniteElement& elem)
{
  fem::FESpace u_space(&mesh, &elem, mesh.dim());
  fem::FESpace p_space(&mesh, &elem);

  fem::MixedFESpace space;
  space.addField(u_space);
  space.addField(p_space);
  space.setup();
  return space;
}

Point3 selectorCenter(const fem::Mesh& mesh, const BoundarySelector& sel)
{
  if (!sel.name.empty())
  {
    return boundaryCenter(mesh, sel.name);
  }
  return boundaryCenter(mesh, sel.physical);
}

Vector<Index> gaugeDofs(const fem::MixedFESpace& space,
                        const BoundarySelector&  sel)
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

fem::DirichletControl makeVelocityControl(
    const fem::MixedFESpace& space,
    const BoundarySelector&  sel)
{
  if (!sel.name.empty())
  {
    return makeVelocityControl(space, sel.name);
  }
  return makeVelocityControl(space, sel.physical);
}

} // namespace femx::model::ns
