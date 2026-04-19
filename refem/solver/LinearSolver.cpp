#include <stdexcept>

#include <refem/solver/LinearSolver.hpp>
#include <refem/solver/LinearSolverImpl.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>

namespace refem
{

LinearSolver::LinearSolver(SolverBackend backend)
{
  switch (backend)
  {
  case SolverBackend::ReSolve:
    throw std::runtime_error(
        "ReSolve LinearSolver requires a user-provided workspace");

  case SolverBackend::Eigen:
    throw std::runtime_error("Eigen linear solver backend is not supported yet");

  case SolverBackend::PETSc:
    throw std::runtime_error("PETSc linear solver backend is not supported yet");

  default:
    throw std::runtime_error("Unknown linear solver backend");
  }
}

LinearSolver::LinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace)
  : impl_(std::make_unique<ReSolveLinearSolver>(workspace))
{
}

LinearSolver::LinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace,
                           ReSolveOptions               options)
  : impl_(std::make_unique<ReSolveLinearSolver>(workspace,
                                                std::move(options)))
{
}

LinearSolver::LinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace)
  : impl_(std::make_unique<ReSolveLinearSolver>(workspace))
{
}

LinearSolver::LinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace,
                           ReSolveOptions                options)
  : impl_(std::make_unique<ReSolveLinearSolver>(workspace,
                                                std::move(options)))
{
}

LinearSolver::LinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace)
  : impl_(std::make_unique<ReSolveLinearSolver>(workspace))
{
}

LinearSolver::LinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace,
                           ReSolveOptions               options)
  : impl_(std::make_unique<ReSolveLinearSolver>(workspace,
                                                std::move(options)))
{
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
