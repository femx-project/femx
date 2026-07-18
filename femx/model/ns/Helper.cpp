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

Array<Index> gaugeDofs(const fem::MixedFESpace& space,
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

  Array<Index> dofs(1);
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

void splitStateFields(HostConstVectorView      state,
                      const fem::MixedFESpace& space,
                      HostVector&              ux,
                      HostVector&              uy,
                      HostVector&              uz,
                      HostVector&              pressure)
{
  const fem::Mesh& mesh      = space.mesh();
  const Index      num_nodes = mesh.numNodes();
  if (state.size() != space.numDofs())
  {
    throw std::runtime_error(
        "Navier-Stokes state size does not match the finite-element space");
  }
  if (ux.size() != num_nodes || uy.size() != num_nodes
      || uz.size() != num_nodes || pressure.size() != num_nodes)
  {
    throw std::runtime_error(
        "Navier-Stokes nodal field output size does not match the mesh");
  }

  const auto  velocity       = space.field(0);
  const auto  pressure_field = space.field(1);
  const Index num_comp       = velocity.numComponents();
  for (Index node = 0; node < num_nodes; ++node)
  {
    ux[node]       = state[velocity.globalDof(node, 0)];
    uy[node]       = num_comp > 1
                         ? state[velocity.globalDof(node, 1)]
                         : 0.0;
    uz[node]       = num_comp > 2
                         ? state[velocity.globalDof(node, 2)]
                         : 0.0;
    pressure[node] = state[pressure_field.globalDof(node)];
  }
}

} // namespace femx::model::ns
