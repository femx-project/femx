#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/state/EnsembleBasis.hpp>

namespace femx
{
namespace state
{

EnsembleBasis::EnsembleBasis(HostVector mean, DenseMatrix perturbations)
  : mean_(std::move(mean)),
    perts_(std::move(perturbations))
{
  checkDims();
}

void EnsembleBasis::reset(HostVector mean, DenseMatrix perturbations)
{
  mean_  = std::move(mean);
  perts_ = std::move(perturbations);
  checkDims();
}

Index EnsembleBasis::numPhysicalParams() const
{
  return mean_.size();
}

Index EnsembleBasis::numCoefficients() const
{
  return perts_.cols();
}

const HostVector& EnsembleBasis::mean() const
{
  return mean_;
}

const DenseMatrix& EnsembleBasis::perturbations() const
{
  return perts_;
}

void EnsembleBasis::apply(const HostVector& alpha,
                          HostVector&       out) const
{
  checkAlpha(alpha);
  out = mean_;
  CpuContext                ctx;
  linalg::HostMatrixHandler mat_handler(ctx);
  mat_handler.matvec(perts_.view(), alpha.view(), out.view(), 1.0, 1.0);
}

void EnsembleBasis::applyT(const HostVector& grad,
                           HostVector&       out) const
{
  checkPhysical(grad);
  out.resize(numCoefficients());
  CpuContext                ctx;
  linalg::HostMatrixHandler mat_handler(ctx);
  mat_handler.matvecT(perts_.view(), grad.view(), out.view());
}

void EnsembleBasis::checkDims() const
{
  require(perts_.rows() >= 0 && perts_.cols() >= 0
              && perts_.rows() == mean_.size(),
          "EnsembleBasis received inconsistent dimensions");
}

void EnsembleBasis::checkAlpha(const HostVector& alpha) const
{
  checkDims();
  require(alpha.size() == numCoefficients(),
          "EnsembleBasis coefficient size mismatch");
}

void EnsembleBasis::checkPhysical(const HostVector& val) const
{
  checkDims();
  require(val.size() == numPhysicalParams(),
          "EnsembleBasis physical vector size mismatch");
}

} // namespace state
} // namespace femx
