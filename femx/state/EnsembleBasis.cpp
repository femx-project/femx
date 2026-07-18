#include <stdexcept>
#include <utility>

#include <femx/state/EnsembleBasis.hpp>

namespace femx
{
namespace state
{

EnsembleBasis::EnsembleBasis(HostVector mean, DenseMatrix perturbations)
  : mean_(std::move(mean)),
    perturbations_(std::move(perturbations))
{
  checkDims();
}

void EnsembleBasis::reset(HostVector mean, DenseMatrix perturbations)
{
  mean_          = std::move(mean);
  perturbations_ = std::move(perturbations);
  checkDims();
}

Index EnsembleBasis::numPhysicalParams() const
{
  return mean_.size();
}

Index EnsembleBasis::numCoefficients() const
{
  return perturbations_.numCols();
}

const HostVector& EnsembleBasis::mean() const
{
  return mean_;
}

const DenseMatrix& EnsembleBasis::perturbations() const
{
  return perturbations_;
}

void EnsembleBasis::apply(const HostVector& alpha,
                          HostVector&       out) const
{
  checkAlpha(alpha);

  HostVector result(numPhysicalParams());
  for (Index i = 0; i < numPhysicalParams(); ++i)
  {
    result[i] = mean_[i];
  }

  for (Index j = 0; j < numCoefficients(); ++j)
  {
    const Real coeff = alpha[j];
    for (Index i = 0; i < numPhysicalParams(); ++i)
    {
      result[i] += perturbations_(i, j) * coeff;
    }
  }

  out = std::move(result);
}

void EnsembleBasis::applyT(const HostVector& grad,
                           HostVector&       out) const
{
  checkPhysical(grad);

  HostVector result(numCoefficients());
  for (Index j = 0; j < numCoefficients(); ++j)
  {
    Real value = 0.0;
    for (Index i = 0; i < numPhysicalParams(); ++i)
    {
      value += perturbations_(i, j) * grad[i];
    }
    result[j] = value;
  }

  out = std::move(result);
}

void EnsembleBasis::checkDims() const
{
  if (perturbations_.numRows() < 0 || perturbations_.numCols() < 0
      || perturbations_.numRows() != mean_.size())
  {
    throw std::runtime_error("EnsembleBasis received inconsistent dimensions");
  }
}

void EnsembleBasis::checkAlpha(const HostVector& alpha) const
{
  checkDims();
  if (alpha.size() != numCoefficients())
  {
    throw std::runtime_error("EnsembleBasis coefficient size mismatch");
  }
}

void EnsembleBasis::checkPhysical(const HostVector& value) const
{
  checkDims();
  if (value.size() != numPhysicalParams())
  {
    throw std::runtime_error("EnsembleBasis physical vector size mismatch");
  }
}

} // namespace state
} // namespace femx
