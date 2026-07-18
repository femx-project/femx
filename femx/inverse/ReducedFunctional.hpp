#pragma once

#include <femx/common/Types.hpp>
#include <femx/state/Residual.hpp>

namespace femx
{
namespace linalg
{
class LinearSolver;
} // namespace linalg

namespace state
{
class Linearization;
class StateSolver;
} // namespace state

namespace inverse
{

class Objective;

/**
 * @brief Reduced objective F(m) = J(u(m), m) using an adjoint gradient.
 *
 * ReducedFunctional owns the high-level optimization operation: solve the
 * state equation for a parameter vector, evaluate the objective, and compute
 * the reduced gradient through an adjoint solve.
 */
class ReducedFunctional
{
public:
  ReducedFunctional(const state::Residual& problem,
                    const Objective&       obj,
                    state::StateSolver&    state_solver,
                    state::Linearization&  lin,
                    linalg::LinearSolver&  adj_lin_solver);

  Index numParams() const;

  /** @brief Evaluate F(prm). */
  Real value(const HostVector& prm);

  /** @brief Compute the reduced gradient dF/dm. */
  void grad(const HostVector& prm, HostVector& out);

  /** @brief Evaluate F(prm) and dF/dm in one state/adjoint pass. */
  Real valueGrad(const HostVector& prm, HostVector& grad_out);

private:
  void checkDims() const;

  void gradAt(const HostVector& state,
              const HostVector& prm,
              HostVector&       out);

  static void checkSize(const HostVector& value, Index exp);

private:
  const state::Residual& problem_;
  const Objective&       obj_;
  state::StateSolver&    state_solver_;
  state::Linearization&  lin_;
  linalg::LinearSolver&  adj_lin_solver_;
  state::Dimensions      dims_;
};

} // namespace inverse
} // namespace femx
