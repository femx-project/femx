#include <refem/fe/ElementValues.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/forms/integrators/DiffusionIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>

namespace refem
{

void DiffusionIntegrator::assemble(const ElementValues& values,
                                   DenseMatrix&         Ke) const
{
  const index_type ndofs = values.numDofs();
  const index_type dim   = values.dim();
  const index_type nq    = values.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const real_type wJ = values.weight(q) * values.detJ(q);
    const auto      dNdx = values.dNdx(q);
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
