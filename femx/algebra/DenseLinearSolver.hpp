#pragma once

#include <vector>

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearOperator.hpp>
#include <femx/algebra/LinearSolver.hpp>

namespace femx
{
namespace algebra
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

} // namespace algebra
} // namespace femx
