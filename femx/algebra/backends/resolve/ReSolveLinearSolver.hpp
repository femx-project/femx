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
#include <femx/algebra/LinearSolver.hpp>

namespace femx
{
class SparseMatrix;
template <typename T>
class Vector;

namespace algebra
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

  Index max_its  = 1000;
  Index restart  = 200;
  Real  rtol     = 1.0e-12;
  bool  flexible = true;
};

class ReSolveLinearSolver final : public LinearSolver
{
public:
  /** @brief Create a ReSolve linear solver for the given work. */
  explicit ReSolveLinearSolver(WorkspaceType workspace_type);

  /** @brief Create a ReSolve linear solver with explicit options. */
  ReSolveLinearSolver(WorkspaceType  workspace_type,
                      ReSolveOptions options);

  /** @brief Destroy the solver and owned ReSolve resources. */
  ~ReSolveLinearSolver() override;

  /** @brief Solve op x = rhs for a SparseMatrixOperator-backed operator. */
  void solve(const LinearOperator& op,
             const Vector<Real>&   rhs,
             Vector<Real>&         out) override;

  /** @brief Solve op^T x = rhs for a SparseMatrixOperator-backed operator. */
  void solveT(const LinearOperator& op,
              const Vector<Real>&   rhs,
              Vector<Real>&         out) override;

  /** @brief Set the system matrix used by subsequent solves. */
  void setOperator(const SparseMatrix& A);

  /** @brief Select or update the preconditioner method. */
  void setPreconditioner(const std::string& method);

  /** @brief Solve A x = b using the current operator. */
  void solve(const Vector<Real>& b, Vector<Real>& x);

private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace algebra
} // namespace femx
