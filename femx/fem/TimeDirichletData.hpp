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
  Vector<Index> dofs;          ///< Fixed state dofs.
  Vector<Real>  values;        ///< Step-major values with shape (steps, dofs).
  Vector<Real>  initial_state; ///< State at time zero with boundary values applied.
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
    Index                    num_states,
    Index                    steps,
    Real                     dt,
    const DirichletBCAtTime& bc_at_time);

} // namespace femx::fem
