#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/forms/integrators/DiffusionIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>

namespace refem
{

void DiffusionIntegrator::assemble(const ElementValues& ev,
                                   DenseMatrix&         Ke) const
{
  const index_type ndofs = ev.numDofs();
  const index_type dim   = ev.dim();
  const index_type nq    = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const real_type wJ   = ev.weight(q) * ev.detJ(q);
    const auto      dNdx = ev.dNdx(q);
    for (index_type i = 0; i < ndofs; ++i)
    {
      for (index_type j = 0; j < ndofs; ++j)
      {
        real_type grad_dot = 0.0;
        for (index_type d = 0; d < dim; ++d)
        {
          grad_dot += dNdx(i, d) * dNdx(j, d);
        }
        Ke(i, j) += kappa_ * grad_dot * wJ;
      }
    }
  }
}

} // namespace refem
