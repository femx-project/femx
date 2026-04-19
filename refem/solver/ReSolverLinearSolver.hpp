#pragma once

#include <memory>
#include <string>

#include <refem/solver/LinearSolverImpl.hpp>

namespace ReSolve
{
class LinAlgWorkspaceCUDA;
class LinAlgWorkspaceCpu;
class LinAlgWorkspaceHIP;
}

namespace refem
{

class SparseMatrix;
class Vector;

class ReSolveLinearSolver final : public LinearSolverImpl
{
public:
  explicit ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace);
  explicit ReSolveLinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace);
  explicit ReSolveLinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace);
  ~ReSolveLinearSolver() override;

  void setOperator(const SparseMatrix& A) override;
  void setPreconditioner(const std::string& method) override;
  void solve(const Vector& b, Vector& x) override;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace refem
