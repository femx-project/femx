#pragma once

#include <memory>
#include <string>

namespace ReSolve
{
class LinAlgWorkspaceCUDA;
class LinAlgWorkspaceCpu;
class LinAlgWorkspaceHIP;
} // namespace ReSolve

namespace refem
{

class SparseMatrix;
class Vector;
class LinearSolverImpl;
struct ReSolveOptions;

enum class SolverBackend
{
  ReSolve,
  Eigen,
  PETSc
};

class LinearSolver
{
public:
  explicit LinearSolver(SolverBackend backend = SolverBackend::ReSolve);
  explicit LinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace);
  LinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace,
               ReSolveOptions               options);

  explicit LinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace);
  LinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace,
               ReSolveOptions                options);

  explicit LinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace);
  LinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace,
               ReSolveOptions               options);

  ~LinearSolver();

  void setOperator(const SparseMatrix& A);
  void setPreconditioner(const std::string& method);
  void solve(const Vector& b, Vector& x);

private:
  std::unique_ptr<LinearSolverImpl> impl_;
};

} // namespace refem
