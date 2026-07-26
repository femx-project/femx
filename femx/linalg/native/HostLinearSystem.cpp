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
  : jacobian_(ctx_), solver_(std::move(solver))
{
  require(solver_ != nullptr,
          "HostLinearSystem requires a linear solver");
}

HostLinearSystem::~HostLinearSystem() = default;

Context<MemorySpace::Host>& HostLinearSystem::context() noexcept
{
  return ctx_;
}

Jacobian<MemorySpace::Host>& HostLinearSystem::jacobian() noexcept
{
  return jacobian_;
}

void HostLinearSystem::solve(ConstView rhs, Vector& solution)
{
  ctx_.vectors().copy(rhs, rhs_);
  solver_->solve(jacobian_.matrix(), rhs_, solution, ctx_);
}

void HostLinearSystem::solveT(ConstView rhs, Vector& solution)
{
  ctx_.vectors().copy(rhs, rhs_);
  solver_->solveT(jacobian_.matrix(), rhs_, solution, ctx_);
}

} // namespace femx::linalg
