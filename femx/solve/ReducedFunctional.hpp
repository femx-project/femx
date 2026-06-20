#pragma once

#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/solve/Newton.hpp>

namespace femx
{
namespace solve
{

/** @brief Reduced objective F(m) = J(u(m), m) using an adjoint gradient. */
class ReducedFunctional
{
public:
  ReducedFunctional(const problem::Residual&  problem,
                    const problem::Objective& objective,
                    Newton&                   state_solver,
                    algebra::LinearSolver&    adjoint_linear_solver);

  Index numParams() const;

  Real value(const Vector<Real>& prm);

  void grad(const Vector<Real>& prm, Vector<Real>& out);

  Real valueGrad(const Vector<Real>& prm, Vector<Real>& grad_out);

private:
  void checkDims() const;

  void gradAt(const Vector<Real>& state,
              const Vector<Real>& prm,
              Vector<Real>&       out);

  static void resize(Vector<Real>& out, Index size);

private:
  const problem::Residual&  problem_;
  const problem::Objective& objective_;
  Newton&                   state_solver_;
  algebra::LinearSolver&    adjoint_linear_solver_;
  problem::Dimensions       dims_;
};

} // namespace solve
} // namespace femx
