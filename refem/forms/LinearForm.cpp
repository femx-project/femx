#include <stdexcept>

#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/forms/LinearForm.hpp>
#include <refem/forms/integrators/DomainLinearIntegrator.hpp>
#include <refem/linalg/LocalAssembler.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

LinearForm::LinearForm(const FESpace* space)
  : space_(space),
    b_(space->numDofs())
{
  if (space_ == nullptr)
  {
    throw std::runtime_error("Cannot construct LinearForm with null FESpace");
  }
}

void LinearForm::addDomainIntegrator(
    std::unique_ptr<DomainLinearIntegrator> integrator)
{
  if (integrator == nullptr)
  {
    throw std::runtime_error("Cannot add a null DomainLinearIntegrator");
  }

  terms_.push_back(std::move(integrator));
}

void LinearForm::assemble()
{
  const auto& fe   = space_->finiteElement();
  const auto& mesh = space_->mesh();

  GaussQuadrature quadrature =
      GaussQuadrature::make(fe.referenceElement(), 2);

  ElementValues ev(fe, quadrature);
  LocalAssembler assembler(*space_);

  b_.setZero();

  for (index_type ic = 0; ic < mesh.numElems(); ++ic)
  {
    ev.reinit(mesh.cell(ic));

    const index_type ndofs = space_->numDofsPerElem();

    Vector Fe(ndofs);
    Fe.setZero();

    for (const auto& term : terms_)
    {
      term->assemble(ev, Fe);
    }

    assembler.addLocalVector(ic, Fe, b_);
  }
}

const Vector& LinearForm::vector() const
{
  return b_;
}

Vector& LinearForm::vector()
{
  return b_;
}

} // namespace refem
