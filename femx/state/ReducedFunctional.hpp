#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/operator/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/Linearization.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/state/StateSolver.hpp>

namespace femx
{
namespace state
{

/** @brief Reduced objective F(m) = J(u(m), m) using an adjoint gradient. */
class ReducedFunctional
{
public:
  ReducedFunctional(const problem::Residual&  problem,
                    const problem::Objective& obj,
                    StateSolver&              state_solver,
                    problem::Linearization&   lin,
                    linalg::LinearSolver&     adj_lin_solver);

  Index numParams() const;

  Real value(const Vector<Real>& prm);

  void grad(const Vector<Real>& prm, Vector<Real>& out);

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
