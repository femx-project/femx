/**
 * @file ReSolveLinearSolver.h
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 *
 */

#pragma once

#include <memory>
#include <string>

#include <femx/core/Types.hpp>
#include <femx/core/Workspace.hpp>

namespace femx
{
class SparseMatrix;
class Vector;

namespace system
{

struct ReSolveOptions
{
  std::string factor   = "klu";
  std::string refactor = "none";
  std::string solve    = "klu";
  std::string precond  = "none";
  std::string ir       = "none";

  std::string gram_schmidt = "cgs2";
  std::string sketching    = "count";

  index_type max_its  = 1000;
  index_type restart  = 200;
  real_type  rtol     = 1.0e-12;
  bool       flexible = true;
};

class ReSolveLinearSolver final
{
public:
  /** @brief Create a ReSolve linear solver for the given work. */
  explicit ReSolveLinearSolver(WorkspaceType workspace_type);

  /** @brief Create a ReSolve linear solver with explicit options. */
  ReSolveLinearSolver(WorkspaceType  workspace_type,
                      ReSolveOptions options);

  /** @brief Destroy the solver and owned ReSolve resources. */
  ~ReSolveLinearSolver();

  /** @brief Set the system matrix used by subsequent solves. */
  void setOperator(const SparseMatrix& A);

  /** @brief Select or update the preconditioner method. */
  void setPreconditioner(const std::string& method);

  /** @brief Solve A x = b using the current operator. */
  void solve(const Vector& b, Vector& x);

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace system
} // namespace femx
