#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/LinearSolver.hpp>

namespace femx
{
template <typename T>
class Vector;

namespace linalg
{

class LinearOperator;

/**
 * @brief Dense fallback solver that samples a LinearOperator into a matrix.
 *
 * DenseLinearSolver is intended for small problems and tests where a direct
 * dense solve is useful as a portable fallback.
 */
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
              Vector<Real>&         mat) const;

  void solveDense(Vector<Real>        mat,
                  const Vector<Real>& rhs,
                  Vector<Real>&       out,
                  Index               size) const;

  static Index entry(Index row, Index col, Index size);

private:
  Real pivot_tolerance_{1.0e-14};
};

} // namespace linalg
} // namespace femx
