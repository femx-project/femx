#include <algorithm>
#include <stdexcept>

#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/forms/BilinearForm.hpp>
#include <refem/forms/integrators/DomainIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>

namespace refem
{

BilinearForm::BilinearForm(const FESpace* space)
  : space_(space),
    pattern_(*space_),
    K_(pattern_)
{
}

void BilinearForm::addDomainIntegrator(
    std::unique_ptr<DomainIntegrator> integrator)
{
  if (integrator == nullptr)
  {
    throw std::runtime_error("Cannot add a null DomainIntegrator");
  }

  terms_.push_back(std::move(integrator));
}

void BilinearForm::assemble()
{
  const auto& fe   = space_->finiteElement();
  const auto& mesh = space_->mesh();

  GaussQuadrature quadrature =
      GaussQuadrature::make(fe.referenceElement(), 2);

  ElementValues values(fe, quadrature);

  K_.setZero();

  index_type ic = 0;

  for (const auto& cell : mesh.cells())
  {
    values.reinit(cell);

    const index_type ndofs = pattern_.elemNumDofs(ic);

    DenseMatrix Ke(ndofs, ndofs);
    Ke.setZero();

    for (const auto& term : terms_)
    {
      term->assemble(values, Ke);
    }

    K_.addLocalMatrix(ic, Ke);

    ++ic;
  }
}

const SparseMatrix& BilinearForm::matrix() const
{
  return K_;
}

SparseMatrix& BilinearForm::matrix()
{
  return K_;
}

} // namespace refem
