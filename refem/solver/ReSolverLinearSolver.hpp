#pragma once

#include <memory>
#include <string>

#include <refem/common/Types.hpp>
#include <refem/solver/LinearSolverImpl.hpp>
#include <refem/solver/Workspace.hpp>

namespace refem
{

class SparseMatrix;
class Vector;

struct ReSolveOptions
{
  std::string factor  = "klu";
  std::string refactor = "none";
  std::string solve   = "klu";
  std::string precond = "none";
  std::string ir      = "none";

  std::string gram_schmidt = "cgs2";
  std::string sketching    = "count";

  index_type max_iterations      = 1000;
  index_type restart             = 200;
  real_type  relative_tolerance  = 1.0e-12;
  bool       flexible            = true;
};

class ReSolveLinearSolver final : public LinearSolverImpl
{
public:
  explicit ReSolveLinearSolver(WorkspaceType workspace_type);
  ReSolveLinearSolver(WorkspaceType workspace_type,
                      ReSolveOptions options);

  ~ReSolveLinearSolver() override;

  void setOperator(const SparseMatrix& A) override;
  void setPreconditioner(const std::string& method) override;
  void solve(const Vector& b, Vector& x) override;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace refem
