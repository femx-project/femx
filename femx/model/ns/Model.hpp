#pragma once

#include <memory>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/ElementQuadratureData.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/ElementKernel.hpp>
#include <femx/model/ns/FluidProperties.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx::model::ns
{

/**
 * @brief Spatial and temporal finite-element discretization of Navier-Stokes.
 */
class NavierStokesModel
{
public:
  NavierStokesModel(const std::string& path,
                    Index              nstep,
                    Real               dt,
                    FluidProperties    fluid);

  NavierStokesModel(fem::Mesh       mesh,
                    Index           nstep,
                    Real            dt,
                    FluidProperties fluid);

  /** @brief Release the model-owned residual implementation. */
  ~NavierStokesModel();

  NavierStokesModel(const NavierStokesModel&)            = delete;
  NavierStokesModel& operator=(const NavierStokesModel&) = delete;
  NavierStokesModel(NavierStokesModel&&)                 = delete;
  NavierStokesModel& operator=(NavierStokesModel&&)      = delete;

  Index numSteps() const;
  Index numStates() const;
  Real  dt() const;

  const FluidProperties& fluid() const;

  const fem::Mesh& mesh() const;

  const fem::MixedFESpace& space() const;

  const fem::HostGeometry& geometry() const;

  /** @brief Return the Host time residual assembled from the shared row operator. */
  state::HostTimeResidual&       residual();
  const state::HostTimeResidual& residual() const;

  const assembly::HostAssemblyMap& map() const;

  /** @brief Return flattened element values reusable by either backend. */
  const fem::HostElementQuadratureData& data() const;

  /** @brief Return the Host row operator for generic time assembly. */
  HostElementKernel elementKernel() const;

  HostVector<Index> velocityDofs() const;
  HostVector<Index> velocityBoundaryDofs(Index boundary_tag) const;
  HostVector<Index> velocityBoundaryDofs(const std::string& boundary_name) const;

private:
  class Residual;

  Index nstep_{0};
  Real  dt_{0.0};

  fem::Mesh                           mesh_;
  std::unique_ptr<fem::FiniteElement> element_;
  fem::MixedFESpace                   space_;
  fem::HostGeometry                   geometry_;
  FluidProperties                     fluid_;
  fem::HostElementQuadratureData      data_;
  assembly::HostAssemblyMap           map_;
  std::unique_ptr<Residual>           res_;
};

/** @brief Copy the parameter-free physics residual and add Device constraints. */
std::unique_ptr<state::DeviceTimeResidual> makeDeviceTimeResidual(
    const NavierStokesModel& model,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state = {});

} // namespace femx::model::ns
