#include <stdexcept>
#include <utility>

#include <refem/solver/LinearSolver.hpp>
#include <refem/solver/LinearSolverImpl.hpp>
#include <refem/solver/ReSolveLinearSolver.hpp>

namespace refem
{

LinearSolver::LinearSolver(WorkspaceType workspace_type,
                           SolverBackend backend)
{
  switch (backend)
  {
  case SolverBackend::ReSolve:
    impl_ = std::make_unique<ReSolveLinearSolver>(workspace_type);
    break;

  case SolverBackend::Eigen:
    throw std::runtime_error("Eigen linear solver backend is not supported yet");

  case SolverBackend::PETSc:
    throw std::runtime_error("PETSc linear solver backend is not supported yet");

  default:
    throw std::runtime_error("Unknown linear solver backend");
  }
}

LinearSolver::LinearSolver(WorkspaceType  workspace_type,
                           SolverBackend  backend,
                           ReSolveOptions options)
{
  switch (backend)
  {
  case SolverBackend::ReSolve:
    impl_ = std::make_unique<ReSolveLinearSolver>(workspace_type,
                                                  std::move(options));
    break;

  case SolverBackend::Eigen:
    throw std::runtime_error("Eigen linear solver backend is not supported yet");

  case SolverBackend::PETSc:
    throw std::runtime_error("PETSc linear solver backend is not supported yet");

  default:
    throw std::runtime_error("Unknown linear solver backend");
  }
}

LinearSolver::~LinearSolver() = default;

void LinearSolver::setOperator(const SparseMatrix& A)
{
  impl_->setOperator(A);
}

void LinearSolver::setPreconditioner(const std::string& method)
{
  impl_->setPreconditioner(method);
}

void LinearSolver::solve(const Vector& b, Vector& x)
{
  impl_->solve(b, x);
}

} // namespace refem
