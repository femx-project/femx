#include <refem/fe/ElementValues.hpp>
#include <refem/forms/integrators/MassIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>

namespace refem
{

MassIntegrator::MassIntegrator(real_type coeff)
  : coeff_(coeff)
{
}

void MassIntegrator::assemble(const ElementValues& ev,
                              DenseMatrix&         Ke) const
{
  const index_type ndofs = ev.numDofs();
  const index_type nq    = ev.numQuadraturePoints();

  for (index_type q = 0; q < nq; ++q)
  {
    const auto      N  = ev.N(q);
    const real_type wJ = ev.JxW(q);

    for (index_type i = 0; i < ndofs; ++i)
    {
      for (index_type j = 0; j < ndofs; ++j)
      {
        Ke(i, j) += coeff_ * N[i] * N[j] * wJ;
      }
    }
  }
}

} // namespace refem
