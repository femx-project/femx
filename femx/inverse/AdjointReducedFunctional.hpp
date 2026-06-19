#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/eq/StateSolver.hpp>
#include <femx/inverse/AdjointSolver.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Reduced objective using one state solve and one adjoint solve. */
class AdjointReducedFunctional : public ReducedFunctional
{
public:
  AdjointReducedFunctional(eq::StateSolver&            state_solver,
                           AdjointSolver&              adj_solver,
                           const eq::ResidualEquation& eq,
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
  eq::StateSolver&            state_solver_;
  AdjointSolver&              adj_solver_;
  const eq::ResidualEquation& eq_;
  const ObjectiveFunctional&  obj_;
};

} // namespace inverse
} // namespace femx
