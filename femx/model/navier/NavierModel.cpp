#include "NavierModel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>
#include <femx/model/navier/NavierElementKernel.hpp>

namespace femx::model::navier
{
namespace
{

constexpr Index kQuadratureOrder = 2;

std::unique_ptr<fem::FiniteElement> makeElement(const fem::Mesh& mesh)
{
  require(mesh.numElems() > 0, "Mesh has no elements");

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
  throw std::runtime_error("Unsupported Navier-Stokes mesh element type");
}

fem::MixedFESpace makeSpace(fem::Mesh& mesh, fem::FiniteElement& elem)
{
  fem::FESpace velocity_space(&mesh, &elem, mesh.dim());
  fem::FESpace pressure_space(&mesh, &elem);

  fem::MixedFESpace space;
  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();
  return space;
}

void requireValidModelOptions(Index                  num_steps,
                              Real                   dt,
                              const FluidProperties& fluid)
{
  require(num_steps > 0,
          "Navier-Stokes model requires a positive number of time steps");
  require(dt > 0.0 && std::isfinite(dt),
          "Navier-Stokes model requires a positive finite time step");
  require(std::isfinite(fluid.rho) && fluid.rho > 0.0,
          "Navier-Stokes model requires positive finite density");
  require(std::isfinite(fluid.mu) && fluid.mu > 0.0,
          "Navier-Stokes model requires positive finite viscosity");
}

fem::Mesh validatedModelMesh(fem::Mesh              mesh,
                             Index                  num_steps,
                             Real                   dt,
                             const FluidProperties& fluid)
{
  requireValidModelOptions(num_steps, dt, fluid);
  return mesh;
}

fem::Mesh readModelMesh(const std::string&     path,
                        Index                  num_steps,
                        Real                   dt,
                        const FluidProperties& fluid)
{
  requireValidModelOptions(num_steps, dt, fluid);
  require(!path.empty(), "Navier-Stokes model mesh file is required");
  return fem::GmshReader::read(path);
}

fem::GaussQuadrature makeVelocityQuadrature(
    const fem::MixedFESpace& space)
{
  return fem::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadratureOrder);
}

fem::HostElementQuadData makeNavierElementData(
    const fem::MixedFESpace& space)
{
  auto data = fem::makeElementQuadData(
      space.field(0).space(), makeVelocityQuadrature(space));
  const Index num_dofs = (data.dim() + 1) * data.numShapes();
  require(data.numElems() > 0 && data.numQuadraturePoints() > 0
              && data.numShapes() > 0 && data.dim() > 0
              && data.dim() <= kMaxDim
              && data.numQuadraturePoints() <= kMaxNq
              && data.numShapes() <= kMaxNn && num_dofs <= kMaxNd,
          "Navier-Stokes element quadrature data has unsupported dimensions");

  return data;
}

} // namespace

NavierModel::NavierModel(const std::string& path,
                         Index              num_steps,
                         Real               dt,
                         FluidProperties    fluid)
  : NavierModel(
        readModelMesh(path, num_steps, dt, fluid),
        num_steps,
        dt,
        fluid)
{
}

NavierModel::NavierModel(fem::Mesh       mesh,
                         Index           num_steps,
                         Real            dt,
                         FluidProperties fluid)
  : num_steps_(num_steps),
    dt_(dt),
    mesh_(validatedModelMesh(
        std::move(mesh), num_steps_, dt_, fluid)),
    elem_(makeElement(mesh_)),
    space_(makeSpace(mesh_, *elem_)),
    fluid_(fluid),
    elem_data_(makeNavierElementData(space_)),
    assm_map_(assembly::makeAssemblyMap(space_.dofMap()))
{
}

Index NavierModel::numSteps() const noexcept
{
  return num_steps_;
}

Index NavierModel::numStates() const noexcept
{
  return space_.numDofs();
}

Real NavierModel::dt() const noexcept
{
  return dt_;
}

const FluidProperties& NavierModel::fluid() const noexcept
{
  return fluid_;
}

const fem::Mesh& NavierModel::mesh() const noexcept
{
  return mesh_;
}

const fem::MixedFESpace& NavierModel::space() const noexcept
{
  return space_;
}

const assembly::HostAssemblyMap&
NavierModel::assemblyMap() const noexcept
{
  return assm_map_;
}

const fem::HostElementQuadData&
NavierModel::elementData() const noexcept
{
  return elem_data_;
}

HostVector<Index> NavierModel::velocityDofs() const
{
  const auto  velocity  = space_.field(0);
  const Index num_nodes = mesh_.numNodes();
  const Index num_comps = velocity.numComponents();

  HostVector<Index> dofs;
  dofs.reserve(num_nodes * num_comps);
  for (Index in = 0; in < num_nodes; ++in)
  {
    for (Index ic = 0; ic < num_comps; ++ic)
    {
      dofs.push_back(velocity.globalDof(in, ic));
    }
  }

  return dofs;
}

HostVector<Index> NavierModel::velocityBoundaryDofs(Index boundary_tag) const
{
  return fem::makeVelocityControl(space_, boundary_tag).stateDofs();
}

HostVector<Index> NavierModel::velocityBoundaryDofs(
    const std::string& boundary_name) const
{
  return fem::makeVelocityControl(space_, boundary_name).stateDofs();
}

} // namespace femx::model::navier
