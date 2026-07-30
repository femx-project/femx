#pragma once

#include "Config.hpp"
#include <femx/common/Vector.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/model/navier/NavierModel.hpp>

namespace femx::apps::navier
{

/** @brief Own the model, constraints, and initial state for a forward run. */
class NavierProblem
{
public:
  /**
   * @brief Construct a forward problem from application configuration.
   *
   * @param[in] prm - Validated application configuration.
   */
  explicit NavierProblem(const Config& prm);

  NavierProblem(const NavierProblem&)            = delete;
  NavierProblem& operator=(const NavierProblem&) = delete;
  NavierProblem(NavierProblem&&)                 = delete;
  NavierProblem& operator=(NavierProblem&&)      = delete;

  /** @brief Return the Navier-Stokes discretization. */
  const model::navier::NavierModel& model() const noexcept;

  /** @brief Return the fixed boundary data. */
  const fem::TimeDirichletData& boundaryData() const noexcept;

  /** @brief Return the boundary control map. */
  const fem::HostControlMap& controlMap() const noexcept;

  /** @brief Return the initial state. */
  const HostVector<Real>& initialState() const noexcept;

private:
  model::navier::NavierModel model_;         ///< Navier-Stokes discretization.
  fem::TimeDirichletData     boundary_data_; ///< Time-dependent boundary values.
  fem::HostControlMap        ctr_map_;       ///< Boundary constraint mapping.
};

} // namespace femx::apps::navier
