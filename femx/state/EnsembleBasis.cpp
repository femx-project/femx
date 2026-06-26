#include <stdexcept>
#include <utility>

#include <femx/state/EnsembleBasis.hpp>

using namespace std;

namespace femx
{
namespace state
{

EnsembleBasis::EnsembleBasis(Vector<Real> mean, DenseMatrix perturbations)
  : mean_(std::move(mean)),
    perturbations_(std::move(perturbations))
{
  checkDims();
}

void EnsembleBasis::reset(Vector<Real> mean, DenseMatrix perturbations)
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
  return perturbations_.cols();
}

const Vector<Real>& EnsembleBasis::mean() const
{
  return mean_;
}

const DenseMatrix& EnsembleBasis::perturbations() const
{
  return perturbations_;
}

void EnsembleBasis::apply(const Vector<Real>& alpha,
                          Vector<Real>&       out) const
{
  checkAlpha(alpha);

  Vector<Real> result(numPhysicalParams());
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

void EnsembleBasis::applyT(const Vector<Real>& grad,
                           Vector<Real>&       out) const
{
  checkPhysical(grad);

  Vector<Real> result(numCoefficients());
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
  if (perturbations_.rows() < 0 || perturbations_.cols() < 0
      || perturbations_.rows() != mean_.size())
  {
    throw runtime_error("EnsembleBasis received inconsistent dimensions");
  }
}

void EnsembleBasis::checkAlpha(const Vector<Real>& alpha) const
{
  checkDims();
  if (alpha.size() != numCoefficients())
  {
    throw runtime_error("EnsembleBasis coefficient size mismatch");
  }
}

void EnsembleBasis::checkPhysical(const Vector<Real>& value) const
{
  checkDims();
  if (value.size() != numPhysicalParams())
  {
    throw runtime_error("EnsembleBasis physical vector size mismatch");
  }
}

} // namespace state
} // namespace femx
