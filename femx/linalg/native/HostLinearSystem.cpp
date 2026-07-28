#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>
#include <femx/linalg/native/HostLinearSystem.hpp>

namespace femx::linalg
{

HostLinearSystem::HostLinearSystem()
  : HostLinearSystem(std::make_unique<DenseLinearSolver>())
{
}

HostLinearSystem::HostLinearSystem(
    std::unique_ptr<LinearSolver<MemorySpace::Host>> solver)
  : mat_(ctx_), solver_(std::move(solver))
{
  require(solver_ != nullptr,
          "HostLinearSystem requires a linear solver");
}

HostLinearSystem::~HostLinearSystem() = default;

Context<MemorySpace::Host>& HostLinearSystem::context() noexcept
{
  return ctx_;
}

SystemMatrix<MemorySpace::Host>& HostLinearSystem::matrix() noexcept
{
  return mat_;
}

void HostLinearSystem::solve(ConstView rhs, Vector& solution)
{
  ctx_.vectorHandler().copy(rhs, rhs_);
  solver_->solve(mat_.matrix(), rhs_, solution, ctx_);
}

void HostLinearSystem::solveT(ConstView rhs, Vector& solution)
{
  ctx_.vectorHandler().copy(rhs, rhs_);
  solver_->solveT(mat_.matrix(), rhs_, solution, ctx_);
}

} // namespace femx::linalg
