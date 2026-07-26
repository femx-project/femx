#pragma once

#include <memory>
#include <string>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ElementQuadData.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/FluidProperties.hpp>

namespace femx::model::ns
{

/**
 * @brief Spatial and temporal finite-element discretization of Navier-Stokes.
 */
class NavierStokesModel
{
public:
  /**
   * @brief Read a mesh and construct the Navier-Stokes discretization.
   *
   * @param[in] path - Mesh file path.
   * @param[in] num_steps - Number of time steps.
   * @param[in] dt - Time-step size.
   * @param[in] fluid - Fluid properties.
   */
  NavierStokesModel(const std::string& path,
                    Index              num_steps,
                    Real               dt,
                    FluidProperties    fluid);

  /**
   * @brief Construct the Navier-Stokes discretization on an owned mesh.
   *
   * @param[in] mesh - Owned finite-element mesh.
   * @param[in] num_steps - Number of time steps.
   * @param[in] dt - Time-step size.
   * @param[in] fluid - Fluid properties.
   */
  NavierStokesModel(fem::Mesh       mesh,
                    Index           num_steps,
                    Real            dt,
                    FluidProperties fluid);

  NavierStokesModel(const NavierStokesModel&)            = delete;
  NavierStokesModel& operator=(const NavierStokesModel&) = delete;
  NavierStokesModel(NavierStokesModel&&)                 = delete;
  NavierStokesModel& operator=(NavierStokesModel&&)      = delete;

  /** @brief Return the number of time steps. */
  Index numSteps() const noexcept;
  /** @brief Return the number of algebraic states. */
  Index numStates() const noexcept;
  /** @brief Return the time-step size. */
  Real  dt() const noexcept;

  /** @brief Return the fluid properties. */
  const FluidProperties& fluid() const noexcept;

  /** @brief Return the finite-element mesh. */
  const fem::Mesh& mesh() const noexcept;

  /** @brief Return the mixed velocity-pressure finite-element space. */
  const fem::MixedFESpace& space() const noexcept;

  /** @brief Return the element assembly map. */
  const assembly::HostAssemblyMap& assemblyMap() const noexcept;

  /** @brief Return flattened element values reusable in either memory space. */
  const fem::HostElementQuadData& elementData() const noexcept;

  /** @brief Return all velocity state degrees of freedom. */
  HostVector<Index> velocityDofs() const;
  /**
   * @brief Return velocity degrees of freedom on a boundary tag.
   *
   * @param[in] boundary_tag - Mesh boundary tag.
   * @return Velocity degrees of freedom on the boundary.
   */
  HostVector<Index> velocityBoundaryDofs(Index boundary_tag) const;

  /**
   * @brief Return velocity degrees of freedom on a named boundary.
   *
   * @param[in] boundary_name - Mesh boundary name.
   * @return Velocity degrees of freedom on the boundary.
   */
  HostVector<Index> velocityBoundaryDofs(const std::string& boundary_name) const;

private:
  Index num_steps_{0}; ///< Number of time steps.
  Real  dt_{0.0};      ///< Time-step size.

  fem::Mesh                           mesh_;      ///< Finite-element mesh.
  std::unique_ptr<fem::FiniteElement> elem_;      ///< Finite element.
  fem::MixedFESpace                   space_;     ///< Velocity-pressure space.
  FluidProperties                     fluid_;     ///< Fluid properties.
  fem::HostElementQuadData      elem_data_; ///< Host integration data.
  assembly::HostAssemblyMap           assm_map_;  ///< Element assembly mapping.
};

} // namespace femx::model::ns
