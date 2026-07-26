#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>

namespace femx::linalg
{

CudaLinearSystem::CudaLinearSystem(
    std::unique_ptr<LinearSolver<MemorySpace::Device>> solver)
  : jacobian_(ctx_), solver_(std::move(solver))
{
  require(solver_ != nullptr,
          "CudaLinearSystem requires a Device linear solver");
}

CudaLinearSystem::~CudaLinearSystem() = default;

Context<MemorySpace::Device>& CudaLinearSystem::context() noexcept
{
  return ctx_;
}

Jacobian<MemorySpace::Device>& CudaLinearSystem::jacobian() noexcept
{
  return jacobian_;
}

void CudaLinearSystem::solve(ConstView rhs, Vector& solution)
{
  ctx_.vectors().copy(rhs, rhs_);
  solver_->solve(jacobian_.matrix(), rhs_, solution, ctx_);
}

void CudaLinearSystem::solveT(ConstView rhs, Vector& solution)
{
  ctx_.vectors().copy(rhs, rhs_);
  solver_->solveT(jacobian_.matrix(), rhs_, solution, ctx_);
}

} // namespace femx::linalg
