#pragma once

#include <memory>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/Components.hpp>
#include <femx/model/ns/Config.hpp>
#include <femx/state/TimeResidual.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScBackend.hpp>
#endif

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
                    FluidParams        fluid);

  NavierStokesModel(fem::Mesh   mesh,
                    Index       nstep,
                    Real        dt,
                    FluidParams fluid);

  /** @brief Release the model-owned residual implementation. */
  ~NavierStokesModel();

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

  /** @brief Return the Host time residual assembled from the shared row operator. */
  state::HostTimeResidual&       residual();
  const state::HostTimeResidual& residual() const;

  /** @brief Restrict Host assembly to the half-open element range. */
  void setElemRange(Index ie_begin, Index ie_end);

  const assembly::HostAssemblyMap& map() const;

  /** @brief Return flattened element values reusable by either backend. */
  const HostNavierData& data() const;

  /** @brief Return the Host row operator for generic time assembly. */
  NavierOperator<MemorySpace::Host> op() const;

  Array<Index> velocityDofs() const;
  Array<Index> velocityBoundaryDofs(Index boundary_tag) const;
  Array<Index> velocityBoundaryDofs(const std::string& boundary_name) const;

private:
  class Residual;

#if defined(FEMX_HAS_PETSC)
  friend std::unique_ptr<state::TimeResidual<linalg::PetscBackend>>
  makePetscTimeResidual(const NavierStokesModel& model);
#endif

  Index nstep_{0};
  Real  dt_{0.0};

  fem::Mesh                           mesh_;
  std::unique_ptr<fem::FiniteElement> element_;
  fem::MixedFESpace                   space_;
  fem::HostGeometry                   geometry_;
  FluidParams                         fluid_;
  HostNavierData                      data_;
  assembly::HostAssemblyMap           map_;
  std::unique_ptr<Residual>           res_;
  Index                               ie_begin_{0};
  Index                               ie_end_{0};
};

/** @brief Copy the parameter-free physics residual and add Device constraints. */
std::unique_ptr<state::DeviceTimeResidual> makeDeviceTimeResidual(
    const NavierStokesModel& model,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state = {});

#if defined(FEMX_HAS_PETSC)
/** @brief Create the PETSc-matrix Navier physics residual. */
std::unique_ptr<state::TimeResidual<linalg::PetscBackend>>
makePetscTimeResidual(const NavierStokesModel& model);
#endif

} // namespace femx::model::ns
