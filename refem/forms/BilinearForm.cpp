#include <algorithm>
#include <stdexcept>

#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/forms/BilinearForm.hpp>
#include <refem/forms/integrators/DomainBilinearIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/LocalAssembler.hpp>

namespace refem
{

BilinearForm::BilinearForm(const FESpace* space)
  : space_(space),
    pattern_(*space_),
    K_(pattern_)
{
}

void BilinearForm::addDomainIntegrator(
    std::unique_ptr<DomainBilinearIntegrator> integrator)
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

  auto quad = GaussQuadrature::make(fe.referenceElement(), 2);

  ElementValues  ev(fe, quad);
  LocalAssembler assembler(*space_, pattern_);

  K_.setZero();

  for (index_type ic = 0; ic < mesh.numElems(); ++ic)
  {
    ev.reinit(mesh.cell(ic));
    const index_type ndofs = pattern_.elemNumDofs(ic);

    DenseMatrix Ke(ndofs, ndofs);
    Ke.setZero();

    for (const auto& term : terms_)
    {
      term->assemble(ev, Ke);
    }
    assembler.addLocalMatrix(ic, Ke, K_);
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
