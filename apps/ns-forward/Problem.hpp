#pragma once

#include "Config.hpp"
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/model/ns/Model.hpp>

namespace femx::apps::ns_forward
{

/** @brief Own the model, constraints, residual, and initial state for a run. */
struct Problem
{
  /**
   * @brief Construct a forward problem from application configuration.
   *
   * @param[in] config - Validated application configuration.
   */
  explicit Problem(const Config& config);

  Problem(const Problem&)            = delete;
  Problem& operator=(const Problem&) = delete;
  Problem(Problem&&)                 = delete;
  Problem& operator=(Problem&&)      = delete;

  model::ns::NavierStokesModel          model;
  fem::TimeDirichletData                boundary_data;
  assembly::HostConstrainedTimeResidual residual;
  HostVector<Real>                      initial_state;
  HostVector<Real>                      parameters;
};

} // namespace femx::apps::ns_forward
