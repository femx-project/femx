#include "NavierStokesModel.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/model/ns/Helper.hpp>

namespace femx::model::ns
{
namespace
{

constexpr Index kQuadratureOrder = 2;

void requireModelParameters(Index              num_steps,
                            Real               dt,
                            const FluidParams& fluid)
{
  if (num_steps <= 0)
  {
    throw std::runtime_error(
        "NavierStokesModel requires a positive number of time steps");
  }
  if (dt <= 0.0 || !std::isfinite(dt))
  {
    throw std::runtime_error(
        "NavierStokesModel requires a positive finite time step");
  }
  if (!std::isfinite(fluid.rho) || fluid.rho <= 0.0)
  {
    throw std::runtime_error(
        "NavierStokesModel requires positive finite density");
  }
  if (!std::isfinite(fluid.mu) || fluid.mu <= 0.0)
  {
    throw std::runtime_error(
        "NavierStokesModel requires positive finite viscosity");
  }
}

fem::Mesh validatedModelMesh(fem::Mesh          mesh,
                             Index              num_steps,
                             Real               dt,
                             const FluidParams& fluid)
{
  requireModelParameters(num_steps, dt, fluid);
  return mesh;
}

fem::Mesh readModelMesh(const std::string& mesh_file,
                        Index              num_steps,
                        Real               dt,
                        const FluidParams& fluid)
{
  requireModelParameters(num_steps, dt, fluid);
  if (mesh_file.empty())
  {
    throw std::runtime_error("NavierStokesModel mesh file is required");
  }
  return fem::GmshReader::read(mesh_file);
}

fem::GaussQuadrature makeVelocityQuadrature(
    const fem::MixedFESpace& space)
{
  return fem::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadratureOrder);
}

NavierKernel makeKernel(const fem::MixedFESpace&    space,
                        const fem::GaussQuadrature& quadrature,
                        const FluidParams&          fluid,
                        Real                        dt)
{
  return makeNavierKernel(space.field(0).space(),
                          quadrature,
                          space.numDofsPerElem(),
                          fluid.rho,
                          fluid.mu,
                          dt);
}

} // namespace

NavierStokesModel::NavierStokesModel(const std::string& mesh_file,
                                     Index              num_steps,
                                     Real               dt,
                                     FluidParams        fluid)
  : NavierStokesModel(
        readModelMesh(mesh_file, num_steps, dt, fluid),
        num_steps,
        dt,
        fluid)
{
}

NavierStokesModel::NavierStokesModel(fem::Mesh   mesh,
                                     Index       num_steps,
                                     Real        dt,
                                     FluidParams fluid)
  : num_steps_(num_steps),
    dt_(dt),
    mesh_(validatedModelMesh(std::move(mesh), num_steps_, dt_, fluid)),
    element_(makeElement(mesh_)),
    space_(makeSpace(mesh_, *element_)),
    geometry_(fem::makeGeometry(mesh_)),
    fluid_(fluid),
    quadrature_(makeVelocityQuadrature(space_)),
    kernel_(makeKernel(space_, quadrature_, fluid_, dt_)),
    res_(num_steps_,
         fem::DofLayout(space_),
         Array<fem::DofLayout>{fem::DofLayout(space_),
                               fem::DofLayout(space_)},
         fem::DofLayout(space_),
         kernel_),
    map_(assembly::makeAssemblyMap(fem::DofLayout(space_)))
{
}

Index NavierStokesModel::numSteps() const
{
  return num_steps_;
}

Index NavierStokesModel::numStates() const
{
  return space_.numDofs();
}

Real NavierStokesModel::dt() const
{
  return dt_;
}

const FluidParams& NavierStokesModel::fluid() const
{
  return fluid_;
}

const fem::Mesh& NavierStokesModel::mesh() const
{
  return mesh_;
}

const fem::MixedFESpace& NavierStokesModel::space() const
{
  return space_;
}

const fem::HostGeometry& NavierStokesModel::geometry() const
{
  return geometry_;
}

assembly::HostTimeResidual& NavierStokesModel::residual()
{
  return res_;
}

const assembly::HostTimeResidual& NavierStokesModel::residual() const
{
  return res_;
}

const assembly::HostAssemblyMap& NavierStokesModel::map() const
{
  return map_;
}

Array<Index> NavierStokesModel::velocityDofs() const
{
  const auto  velocity  = space_.field(0);
  const Index num_nodes = mesh_.numNodes();
  const Index num_comps = velocity.numComponents();

  Array<Index> dofs;
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

Array<Index> NavierStokesModel::velocityBoundaryDofs(
    Index boundary_tag) const
{
  return fem::makeVelocityControl(space_, boundary_tag).stateDofs();
}

Array<Index> NavierStokesModel::velocityBoundaryDofs(
    const std::string& boundary_name) const
{
  return fem::makeVelocityControl(space_, boundary_name).stateDofs();
}

} // namespace femx::model::ns
