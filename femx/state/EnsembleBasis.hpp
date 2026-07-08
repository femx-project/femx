#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace state
{

/**
 * @brief Low-rank parameter basis prm = mean + perturbations * alpha.
 *
 * EnsembleBasis maps ensemble coefficients into physical parameters and
 * projects gradients back to coefficient space.
 */
class EnsembleBasis final
{
public:
  EnsembleBasis() = default;

  EnsembleBasis(Vector<Real> mean, DenseMatrix perturbations);

  void reset(Vector<Real> mean, DenseMatrix perturbations);

  Index numPhysicalParams() const;
  Index numCoefficients() const;

  const Vector<Real>& mean() const;
  const DenseMatrix&  perturbations() const;

  void apply(const Vector<Real>& alpha, Vector<Real>& out) const;
  void applyT(const Vector<Real>& grad, Vector<Real>& out) const;

private:
  void checkDims() const;
  void checkAlpha(const Vector<Real>& alpha) const;
  void checkPhysical(const Vector<Real>& value) const;

private:
  Vector<Real> mean_;
  DenseMatrix  perturbations_;
};

} // namespace state
} // namespace femx
