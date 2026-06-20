#pragma once

// Compatibility reduced-functional implementation. Prefer
// <femx/solve/ReducedObjective.hpp> for new code.
#include <femx/core/Types.hpp>
#include <femx/problem/ResidualEquation.hpp>
#include <femx/solve/StateSolver.hpp>
#include <femx/solve/AdjointSolver.hpp>
#include <femx/problem/ObjectiveFunctional.hpp>
#include <femx/solve/ReducedObjective.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace solve
{

using problem::ObjectiveFunctional;

/** @brief Reduced objective using one state solve and one adjoint solve. */
class AdjointReducedFunctional : public ReducedObjective
{
public:
  AdjointReducedFunctional(solve::StateSolver&            state_solver,
                           AdjointSolver&              adj_solver,
                           const problem::ResidualEquation& eq,
                           const ObjectiveFunctional&  obj);

  Index numParams() const override;

  Real value(const Vector<Real>& prm) override;

  void grad(const Vector<Real>& prm,
            Vector<Real>&       out) override;

  Real valueGrad(const Vector<Real>& prm,
                 Vector<Real>&       grad_out) override;

private:
  void checkDims() const;

  void gradAt(const Vector<Real>& state,
              const Vector<Real>& prm,
              Vector<Real>&       out) const;

private:
  solve::StateSolver&            state_solver_;
  AdjointSolver&              adj_solver_;
  const problem::ResidualEquation& eq_;
  const ObjectiveFunctional&  obj_;
};

} // namespace solve
} // namespace femx
