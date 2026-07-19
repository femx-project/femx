#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Dense.hpp>
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

  EnsembleBasis(HostVector mean, DenseMatrix perturbations);

  void reset(HostVector mean, DenseMatrix perturbations);

  Index numPhysicalParams() const;
  Index numCoefficients() const;

  const HostVector&  mean() const;
  const DenseMatrix& perturbations() const;

  void apply(const HostVector& alpha, HostVector& out) const;
  void applyT(const HostVector& grad, HostVector& out) const;

private:
  void checkDims() const;
  void checkAlpha(const HostVector& alpha) const;
  void checkPhysical(const HostVector& val) const;

private:
  HostVector  mean_;
  DenseMatrix perts_ ;
};

} // namespace state
} // namespace femx
