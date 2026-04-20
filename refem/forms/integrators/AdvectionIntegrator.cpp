#include <stdexcept>

#include <refem/fe/ElementValues.hpp>
#include <refem/forms/integrators/AdvectionIntegrator.hpp>
#include <refem/linalg/DenseMatrix.hpp>

namespace refem
{

AdvectionIntegrator::AdvectionIntegrator(std::vector<real_type> velocity,
                                         real_type              coeff)
  : velocity_(std::move(velocity)),
    coeff_(coeff)
{
}

void AdvectionIntegrator::assemble(const ElementValues& ev,
                                   DenseMatrix&         Ke) const
{
  const index_type dim   = ev.dim();
  const index_type ndofs = ev.numDofs();
  const index_type nq    = ev.numQuadraturePoints();

  if (static_cast<index_type>(velocity_.size()) != nq * dim)
  {
    throw std::runtime_error(
        "AdvectionIntegrator velocity size must be nq * dim");
  }

  for (index_type q = 0; q < nq; ++q)
  {
    const auto      N    = ev.N(q);
    const auto      dNdx = ev.dNdx(q);
    const real_type wJ   = ev.JxW(q);

    for (index_type i = 0; i < ndofs; ++i)
    {
      for (index_type j = 0; j < ndofs; ++j)
      {
        real_type deriv = 0.0;
        for (index_type d = 0; d < dim; ++d)
        {
          deriv += velocity_[static_cast<std::size_t>(q * dim + d)] * dNdx(j, d);
        }
        Ke(i, j) += coeff_ * N[i] * deriv * wJ;
      }
    }
  }
}

} // namespace refem
