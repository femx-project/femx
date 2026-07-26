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

  EnsembleBasis(HostVector<Real> mean, DenseMatrix perturbations);

  void reset(HostVector<Real> mean, DenseMatrix perturbations);

  Index numPhysicalParams() const;
  Index numCoefficients() const;

  const HostVector<Real>& mean() const;
  const DenseMatrix&      perturbations() const;

  void apply(const HostVector<Real>& alpha, HostVector<Real>& out) const;
  void applyT(const HostVector<Real>& grad, HostVector<Real>& out) const;

private:
  void checkDims() const;
  void checkAlpha(const HostVector<Real>& alpha) const;
  void checkPhysical(const HostVector<Real>& val) const;

private:
  HostVector<Real> mean_;
  DenseMatrix      perts_;
};

} // namespace state
} // namespace femx
