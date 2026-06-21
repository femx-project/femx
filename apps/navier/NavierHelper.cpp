#include "NavierHelper.hpp"

#include <stdexcept>

#include <femx/common/Math.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/VelocityProfile.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>

namespace femx::navier
{

std::unique_ptr<FiniteElement> makeElement(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("Mesh has no cells");
  }

  const Cell::Shape shape = mesh.cells().front().shape();
  if (shape == Cell::Shape::Quadrilateral)
  {
    return std::make_unique<LagrangeQuadQ1>();
  }
  if (shape == Cell::Shape::Triangle)
  {
    return std::make_unique<LagrangeTriangleP1>();
  }
  if (shape == Cell::Shape::Tetrahedron)
  {
    return std::make_unique<LagrangeTetrahedronP1>();
  }
  throw std::runtime_error("Unsupported mesh cell type for Navier app");
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

Point3 selectorCenter(const Mesh& mesh, const BoundarySelector& selector)
{
  if (!selector.name.empty())
  {
    return fem::boundaryCenter(mesh, selector.name);
  }
  return fem::boundaryCenter(mesh, selector.physical);
}

Vector<Index> gaugeDofs(const MixedFESpace&     space,
                        const BoundarySelector& selector)
{
  Index      node_out = 0;
  Real       dist_out = 0.0;
  const auto center   = selectorCenter(space.mesh(), selector);
  for (Index node = 0; node < space.mesh().numNodes(); ++node)
  {
    const Real dist = sqDist(space.mesh().node(node), center);
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
    const BoundarySelector& selector)
{
  if (!selector.name.empty())
  {
    return femx::makeVelocityControl(space, selector.name);
  }
  return femx::makeVelocityControl(space, selector.physical);
}

} // namespace femx::navier
