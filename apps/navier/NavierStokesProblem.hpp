#pragma once

#include "Config.hpp"
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/Model.hpp>

namespace femx::apps::navier
{

/** @brief Own the model, constraints, and initial state for a forward run. */
class NavierStokesProblem
{
public:
  /**
   * @brief Construct a forward problem from application configuration.
   *
   * @param[in] prm - Validated application configuration.
   */
  explicit NavierStokesProblem(const Config& prm);

  NavierStokesProblem(const NavierStokesProblem&)            = delete;
  NavierStokesProblem& operator=(const NavierStokesProblem&) = delete;
  NavierStokesProblem(NavierStokesProblem&&)                 = delete;
  NavierStokesProblem& operator=(NavierStokesProblem&&)      = delete;

  /** @brief Return the Navier-Stokes discretization. */
  const model::ns::NavierStokesModel& model() const noexcept;

  /** @brief Return the fixed boundary data. */
  const fem::TimeDirichletData& boundaryData() const noexcept;

  /** @brief Return the boundary control map. */
  const fem::HostControlMap& controlMap() const noexcept;

  /** @brief Return the initial state. */
  const HostVector<Real>& initialState() const noexcept;

private:
  model::ns::NavierStokesModel model_;         ///< Navier-Stokes discretization.
  fem::TimeDirichletData       boundary_data_; ///< Time-dependent boundary values.
  fem::HostControlMap          control_map_;   ///< Boundary constraint mapping.
};

} // namespace femx::apps::navier
