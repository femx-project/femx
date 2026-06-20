#include <petsctao.h>

#include <iomanip>
#include <iostream>

#include "Problem.hpp"
#include <femx/optimize/TaoOptimizer.hpp>
#include <femx/solve/AdjointReducedFunctional.hpp>
#include <femx/solve/MatrixAdjointSolver.hpp>
#include <femx/solve/MatrixNewtonStateSolver.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/backends/petsc/KspLinearSolver.hpp>
#include <femx/algebra/backends/petsc/PETScSystemMatrix.hpp>

using namespace femx;
using namespace femx::examples_inverse_linear_control;

namespace
{

PetscErrorCode runOptimization()
{
  LinearResidualEquation     res_eq;
  algebra::PETScSystemMatrix state_jac;
  algebra::PETScSystemMatrix adj_state_jac;
  algebra::KspLinearSolver   lin_solver;
  lin_solver.options().pc_type     = PCJACOBI;
  lin_solver.options().rtol        = 1.0e-12;
  lin_solver.options().atol        = 1.0e-14;
  lin_solver.options().use_opts_db = true;

  solve::MatrixNewtonStateSolver state_solver(
      res_eq, state_jac, lin_solver);
  solve::MatrixAdjointSolver adj_solver(
      res_eq, adj_state_jac, lin_solver);
  LinearControlObjectiveParts objective_parts;

  solve::AdjointReducedFunctional functional(
      state_solver, adj_solver, res_eq, objective_parts.objective);

  Vector<Real> initial(2);
  initial[0] = 0.05;
  initial[1] = -0.02;

  optimize::TaoOptimizer optimizer(functional);
  optimizer.options().grad_abs_tolerance  = 1.0e-10;
  optimizer.options().grad_rel_tolerance  = 1.0e-10;
  optimizer.options().grad_step_tolerance = 0.0;
  optimizer.options().max_its             = 100;

  optimize::TaoResult result;
  PetscCall(optimizer.solve(initial, result));

  Vector<Real> state;
  state_solver.solve(result.prm, state);

  std::cout << std::setprecision(12);
  std::cout << "example-inverse-linear-control\n";
  printVector("initial prm", initial);
  printVector("prm", result.prm);
  printVector("state", state);
  std::cout << "value = " << result.value << '\n';
  printVector("grad", result.grad);
  std::cout << "grad norm squared = " << result.grad_norm_squared << '\n';
  std::cout << "its = " << result.its << '\n';
  std::cout << "tao reason = " << static_cast<int>(result.reason) << '\n';

  return result.converged() ? PETSC_SUCCESS : PETSC_ERR_NOT_CONVERGED;
}

} // namespace

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  ierr = runOptimization();

  const PetscErrorCode finalize_ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS || finalize_ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  return 0;
}
