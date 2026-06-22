#include <petsctao.h>

#include <iomanip>
#include <iostream>

#include "Problem.hpp"
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/backends/petsc/KspLinearSolver.hpp>
#include <femx/linalg/backends/petsc/PETScSystemMatrix.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/state/AdjointReducedFunctional.hpp>
#include <femx/state/MatrixAdjointSolver.hpp>
#include <femx/state/MatrixNewtonStateSolver.hpp>

using namespace femx;
using namespace femx::examples_inverse_linear_control;

namespace
{

PetscErrorCode runOptimization()
{
  LinearResidualEquation    res_eq;
  linalg::PETScSystemMatrix state_jac;
  linalg::PETScSystemMatrix adj_state_jac;
  linalg::KspLinearSolver   lin_solver;
  lin_solver.opts().pc_type     = PCJACOBI;
  lin_solver.opts().rtol        = 1.0e-12;
  lin_solver.opts().atol        = 1.0e-14;
  lin_solver.opts().use_opts_db = true;

  state::MatrixNewtonStateSolver state_solver(
      res_eq, state_jac, lin_solver);
  state::MatrixAdjointSolver adj_solver(
      res_eq, adj_state_jac, lin_solver);
  LinearControlObjectiveParts obj_parts;

  state::AdjointReducedFunctional fn(
      state_solver, adj_solver, res_eq, obj_parts.obj);

  Vector<Real> initial(2);
  initial[0] = 0.05;
  initial[1] = -0.02;

  opt::TaoOptimizer optimizer(fn);
  optimizer.opts().abs_tol  = 1.0e-10;
  optimizer.opts().rel_tol  = 1.0e-10;
  optimizer.opts().step_tol = 0.0;
  optimizer.opts().max_its  = 100;

  opt::TaoResult result;
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
