#include <petsctao.h>

#include <iomanip>
#include <iostream>

#include "Problem.hpp"
#include <femx/eq/MatrixNewtonStateSolver.hpp>
#include <femx/inverse/AdjointReducedFunctional.hpp>
#include <femx/inverse/MatrixEquationAdjointSolver.hpp>
#include <femx/inverse/petsc/TaoOptimizer.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/petsc/KspLinearSolver.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>

using namespace femx;
using namespace femx::examples_inverse_linear_control;

namespace
{

PetscErrorCode runOptimization()
{
  LinearResidualEquation    residual_equation;
  system::PETScSystemMatrix state_jac;
  system::PETScSystemMatrix adj_state_jac;
  system::KspLinearSolver   lin_solver;
  lin_solver.options().pc_type     = PCJACOBI;
  lin_solver.options().rtol        = 1.0e-12;
  lin_solver.options().atol        = 1.0e-14;
  lin_solver.options().use_opts_db = true;

  equation::MatrixNewtonStateSolver state_solver(
      residual_equation, state_jac, lin_solver);
  inverse::MatrixEquationAdjointSolver adj_solver(
      residual_equation, adj_state_jac, lin_solver);
  LinearControlObjectiveParts objective_parts;

  inverse::AdjointReducedFunctional functional(
      state_solver, adj_solver, residual_equation, objective_parts.objective);

  Vector initial(2);
  initial[0] = 0.05;
  initial[1] = -0.02;

  inverse::TaoOptimizer optimizer(functional);
  optimizer.options().grad_abs_tolerance  = 1.0e-10;
  optimizer.options().grad_rel_tolerance  = 1.0e-10;
  optimizer.options().grad_step_tolerance = 0.0;
  optimizer.options().max_its             = 100;

  inverse::TaoResult result;
  PetscCall(optimizer.solve(initial, result));

  Vector state;
  state_solver.solve(result.params, state);

  std::cout << std::setprecision(12);
  std::cout << "example-inverse-linear-control\n";
  printVector("initial params", initial);
  printVector("params", result.params);
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
