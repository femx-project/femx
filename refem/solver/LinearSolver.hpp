#pragma once

#include <memory>
#include <string>

#include <refem/linalg/LinalgBackend.hpp>

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
class LinearSolverImpl;

class LinearSolver
{
public:
  explicit LinearSolver(LinalgBackend backend = LinalgBackend::ReSolve);
  explicit LinearSolver(ReSolve::LinAlgWorkspaceCpu* workspace);
  explicit LinearSolver(ReSolve::LinAlgWorkspaceCUDA* workspace);
  explicit LinearSolver(ReSolve::LinAlgWorkspaceHIP* workspace);

  ~LinearSolver();

  void setOperator(const SparseMatrix& A);

  void setPreconditioner(const std::string& method);

  void solve(const Vector& b, Vector& x);

private:
  std::unique_ptr<LinearSolverImpl> impl_;
};

} // namespace refem
