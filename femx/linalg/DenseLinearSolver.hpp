#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx
{
namespace linalg
{

/** @brief Dense fallback solver that samples a LinearOperator into a matrix. */
class DenseLinearSolver final : public LinearSolver
{
public:
  explicit DenseLinearSolver(Real pivot_tolerance = 1.0e-14);

  void solve(const LinearOperator& op,
             const Vector<Real>&   rhs,
             Vector<Real>&         out) override;

  void solveT(const LinearOperator& op,
              const Vector<Real>&   rhs,
              Vector<Real>&         out) override;

private:
  void sample(const LinearOperator& op,
              bool                  transpose,
              std::vector<Real>&    mat) const;

  void solveDense(std::vector<Real>   mat,
                  const Vector<Real>& rhs,
                  Vector<Real>&       out,
                  Index               size) const;

  static std::size_t entry(Index row,
                           Index col,
                           Index size);

  static void resize(Vector<Real>& out, Index size);

private:
  Real pivot_tolerance_{1.0e-14};
};

} // namespace linalg
} // namespace femx
