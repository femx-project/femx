#pragma once

#include <functional>

#include <femx/common/Types.hpp>
#include <femx/fem/DirichletBC.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::fem
{

/** Fixed Dirichlet data sampled at every time level. */
struct TimeDirichletData
{
  Array<Index> dofs;       ///< Fixed state dofs.
  HostVector   vals;       ///< Step-major values with shape (steps, dofs).
  HostVector   init_state; ///< State at time zero with boundary values applied.
};

using DirichletBCAtTime =
    std::function<DirichletBC(Real)>;

/**
 * Make fixed boundary data for a time residual.
 *
 * The callback is evaluated at time zero and at `(step + 1) * dt`. Duplicate
 * dofs must carry equal values, and the constrained dof set must remain fixed
 * over time.
 */
TimeDirichletData makeTimeDirichletData(
    Index                    nstate,
    Index                    nstep,
    Real                     dt,
    const DirichletBCAtTime& bc_at_time);

} // namespace femx::fem
