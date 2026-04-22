/**
 * @file ReSolveLinearSolver.h
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 *
 */

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
  std::string factor   = "klu";
  std::string refactor = "none";
  std::string solve    = "klu";
  std::string precond  = "none";
  std::string ir       = "none";

  std::string gram_schmidt = "cgs2";
  std::string sketching    = "count";

  index_type max_iterations     = 1000;
  index_type restart            = 200;
  real_type  relative_tolerance = 1.0e-12;
  bool       flexible           = true;
};

class ReSolveLinearSolver final : public LinearSolverImpl
{
public:
  /**
   * @brief Create a ReSolve linear solver for the given workspace.
   */
  explicit ReSolveLinearSolver(WorkspaceType workspace_type);

  /**
   * @brief Create a ReSolve linear solver with explicit options.
   */
  ReSolveLinearSolver(WorkspaceType  workspace_type,
                      ReSolveOptions options);

  /**
   * @brief Destroy the solver and owned ReSolve resources.
   */
  ~ReSolveLinearSolver() override;

  /**
   * @brief Set the system matrix used by subsequent solves.
   */
  void setOperator(const SparseMatrix& A) override;

  /**
   * @brief Select or update the preconditioner method.
   */
  void setPreconditioner(const std::string& method) override;

  /**
   * @brief Solve A x = b using the current operator.
   */
  void solve(const Vector& b, Vector& x) override;

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace refem
