#include <femx/linalg/petsc/PETScLinearSystem.hpp>

namespace femx::linalg
{

PETScLinearSystem::PETScLinearSystem(MPI_Comm comm)
  : ctx_(comm), mat_(ctx_), solver_(ctx_.comm())
{
}

Context<MemorySpace::Host>& PETScLinearSystem::context() noexcept
{
  return ctx_;
}

SystemMatrix<MemorySpace::Host>& PETScLinearSystem::matrix() noexcept
{
  return mat_;
}

void PETScLinearSystem::solve(ConstView rhs, Vector& solution)
{
  ctx_.vectors().copy(rhs, rhs_);
  solver_.solve(mat_.matrix(), rhs_, solution);
}

void PETScLinearSystem::solveT(ConstView rhs, Vector& solution)
{
  ctx_.vectors().copy(rhs, rhs_);
  solver_.solveT(mat_.matrix(), rhs_, solution);
}

PETScLinearSolver& PETScLinearSystem::solver() noexcept
{
  return solver_;
}

const PETScLinearSolver& PETScLinearSystem::solver() const noexcept
{
  return solver_;
}

} // namespace femx::linalg
