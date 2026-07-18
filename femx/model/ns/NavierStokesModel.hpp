#pragma once

#include <memory>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/HostTimeResidual.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/Config.hpp>
#include <femx/model/ns/Kernel.hpp>

namespace femx::model::ns
{

/**
 * @brief Spatial and temporal finite-element discretization of Navier-Stokes.
 */
class NavierStokesModel
{
public:
  NavierStokesModel(const std::string& mesh_file,
                    Index              num_steps,
                    Real               dt,
                    FluidParams        fluid);

  NavierStokesModel(fem::Mesh   mesh,
                    Index       num_steps,
                    Real        dt,
                    FluidParams fluid);

  NavierStokesModel(const NavierStokesModel&)            = delete;
  NavierStokesModel& operator=(const NavierStokesModel&) = delete;
  NavierStokesModel(NavierStokesModel&&)                 = delete;
  NavierStokesModel& operator=(NavierStokesModel&&)      = delete;

  Index numSteps() const;
  Index numStates() const;
  Real  dt() const;

  const FluidParams& fluid() const;

  const fem::Mesh& mesh() const;

  const fem::MixedFESpace& space() const;

  const fem::HostGeometry& geometry() const;

  assembly::HostTimeResidual&       residual();
  const assembly::HostTimeResidual& residual() const;

  const assembly::HostAssemblyMap& map() const;

  Array<Index> velocityDofs() const;
  Array<Index> velocityBoundaryDofs(Index boundary_tag) const;
  Array<Index> velocityBoundaryDofs(
      const std::string& boundary_name) const;

private:
  Index num_steps_{0};
  Real  dt_{0.0};

  fem::Mesh                           mesh_;
  std::unique_ptr<fem::FiniteElement> element_;
  fem::MixedFESpace                   space_;
  fem::HostGeometry                   geometry_;
  FluidParams                         fluid_;
  fem::GaussQuadrature                quadrature_;
  NavierKernel                        kernel_;
  assembly::HostTimeResidual          res_;
  assembly::HostAssemblyMap           map_;
};

} // namespace femx::model::ns
