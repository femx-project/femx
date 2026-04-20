#pragma once

#include <memory>
#include <string>

#include <refem/solver/Workspace.hpp>

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
  explicit LinearSolver(WorkspaceType workspace_type,
                        SolverBackend backend = SolverBackend::ReSolve);
  LinearSolver(WorkspaceType  workspace_type,
               SolverBackend  backend,
               ReSolveOptions options);

  ~LinearSolver();

  void setOperator(const SparseMatrix& A);
  void setPreconditioner(const std::string& method);
  void solve(const Vector& b, Vector& x);

private:
  std::unique_ptr<LinearSolverImpl> impl_;
};

} // namespace refem
