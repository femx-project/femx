#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/problem/Linearization.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx
{
namespace state
{

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
  ReducedFunctional(const problem::Residual&  problem,
                    const problem::Objective& obj,
                    StateSolver&              state_solver,
                    problem::Linearization&   lin,
                    linalg::LinearSolver&     adj_lin_solver);

  Index numParams() const;

  /** @brief Evaluate F(prm). */
  Real value(const Vector<Real>& prm);

  /** @brief Compute the reduced gradient dF/dm. */
  void grad(const Vector<Real>& prm, Vector<Real>& out);

  /** @brief Evaluate F(prm) and dF/dm in one state/adjoint pass. */
  Real valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);

private:
  void checkDims() const;

  void gradAt(const Vector<Real>& state,
              const Vector<Real>& prm,
              Vector<Real>&       out);

  static void checkSize(const Vector<Real>& value, Index exp);

private:
  const problem::Residual&  problem_;
  const problem::Objective& obj_;
  StateSolver&              state_solver_;
  problem::Linearization&   lin_;
  linalg::LinearSolver&     adj_lin_solver_;
  problem::Dimensions       dims_;
};

} // namespace state
} // namespace femx
