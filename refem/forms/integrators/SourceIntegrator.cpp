#include <refem/fe/ElementValues.hpp>
#include <refem/forms/integrators/SourceIntegrator.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

SourceIntegrator::SourceIntegrator(real_type value)
  : value_(value)
{
}

void SourceIntegrator::assemble(const ElementValues& ev,
                                Vector&              Fe) const
{
  const index_type nq    = ev.numQuadraturePoints();
  const index_type ndofs = ev.numDofs();

  for (index_type q = 0; q < nq; ++q)
  {
    const real_type JxW = ev.JxW(q);
    const auto      N   = ev.N(q);

    for (index_type i = 0; i < ndofs; ++i)
    {
      Fe[i] += value_ * N[i] * JxW;
    }
  }
}

} // namespace refem
