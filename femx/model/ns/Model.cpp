#include "Model.hpp"

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
#include <femx/model/ns/ElementKernel.hpp>

namespace femx::model::ns
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
          "NavierStokesModel requires a positive number of time steps");
  require(dt > 0.0 && std::isfinite(dt),
          "NavierStokesModel requires a positive finite time step");
  require(std::isfinite(fluid.rho) && fluid.rho > 0.0,
          "NavierStokesModel requires positive finite density");
  require(std::isfinite(fluid.mu) && fluid.mu > 0.0,
          "NavierStokesModel requires positive finite viscosity");
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
  require(!path.empty(), "NavierStokesModel mesh file is required");
  return fem::GmshReader::read(path);
}

fem::GaussQuadrature makeVelocityQuadrature(
    const fem::MixedFESpace& space)
{
  return fem::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadratureOrder);
}

fem::HostElementQuadData makeNavierStokesElementData(
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
          "Navier element quadrature data has unsupported dimensions");
  return data;
}

} // namespace

NavierStokesModel::NavierStokesModel(const std::string& path,
                                     Index              num_steps,
                                     Real               dt,
                                     FluidProperties    fluid)
  : NavierStokesModel(
        readModelMesh(path, num_steps, dt, fluid),
        num_steps,
        dt,
        fluid)
{
}

NavierStokesModel::NavierStokesModel(fem::Mesh       mesh,
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
    elem_data_(makeNavierStokesElementData(space_)),
    assm_map_(assembly::makeAssemblyMap(space_.dofMap()))
{
}

Index NavierStokesModel::numSteps() const noexcept
{
  return num_steps_;
}

Index NavierStokesModel::numStates() const noexcept
{
  return space_.numDofs();
}

Real NavierStokesModel::dt() const noexcept
{
  return dt_;
}

const FluidProperties& NavierStokesModel::fluid() const noexcept
{
  return fluid_;
}

const fem::Mesh& NavierStokesModel::mesh() const noexcept
{
  return mesh_;
}

const fem::MixedFESpace& NavierStokesModel::space() const noexcept
{
  return space_;
}

const assembly::HostAssemblyMap&
NavierStokesModel::assemblyMap() const noexcept
{
  return assm_map_;
}

const fem::HostElementQuadData&
NavierStokesModel::elementData() const noexcept
{
  return elem_data_;
}

HostVector<Index> NavierStokesModel::velocityDofs() const
{
  const auto  velocity  = space_.field(0);
  const Index num_nodes = mesh_.numNodes();
  const Index num_comps = velocity.numComponents();

  HostVector<Index> dofs;
  dofs.reserve(num_nodes * num_comps);
  for (Index node = 0; node < num_nodes; ++node)
  {
    for (Index component = 0; component < num_comps; ++component)
    {
      dofs.push_back(velocity.globalDof(node, component));
    }
  }
  return dofs;
}

HostVector<Index> NavierStokesModel::velocityBoundaryDofs(
    Index boundary_tag) const
{
  return fem::makeVelocityControl(space_, boundary_tag).stateDofs();
}

HostVector<Index> NavierStokesModel::velocityBoundaryDofs(
    const std::string& boundary_name) const
{
  return fem::makeVelocityControl(space_, boundary_name).stateDofs();
}

} // namespace femx::model::ns
