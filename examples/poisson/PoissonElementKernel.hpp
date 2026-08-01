#pragma once

#include <femx/assembly/Assembly.hpp>
#include <femx/fem/ElementQuadData.hpp>

namespace femx::examples::poisson
{

/**
 * @brief Evaluate the scalar Laplace operator from element quadrature data.
 */
template <MemorySpace Space>
class PoissonElementKernel
{
public:
  /**
   * @brief Bind the element quadrature data used during assembly.
   *
   * @param[in] data - Shape gradients and weighted Jacobians.
   */
  FEMX_HOST_DEVICE explicit PoissonElementKernel(
      fem::ElementQuadDataView<Space> data)
    : data_(data)
  {
  }

  /**
   * @brief Evaluate one local residual row and its Jacobian row.
   *
   * @param[in]  elem - Element state and metadata.
   * @param[in]  row - Local residual row.
   * @param[out] res - Local residual value.
   * @param[out] jac - Local Jacobian row.
   */
  FEMX_HOST_DEVICE void evalRow(
      const assembly::ElementView<Space>& elem,
      Index                               row,
      Real&                               res,
      VectorView<Space, Real>             jac) const
  {
    res = 0.0;
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = 0.0;
    }

    for (Index iq = 0; iq < data_.numQuadraturePoints(); ++iq)
    {
      for (Index col = 0; col < jac.size(); ++col)
      {
        Real grad_dot = 0.0;
        for (Index id = 0; id < data_.dim(); ++id)
        {
          grad_dot += data_.dNdx(elem.ie, iq, row, id)
                      * data_.dNdx(elem.ie, iq, col, id);
        }
        jac[col] += grad_dot * data_.JxW(elem.ie, iq);
      }
    }

    for (Index col = 0; col < jac.size(); ++col)
    {
      res += jac[col] * elem.state[col];
    }
  }

private:
  fem::ElementQuadDataView<Space> data_; ///< Element integration data.
};

using HostPoissonElementKernel   = PoissonElementKernel<MemorySpace::Host>;
using DevicePoissonElementKernel = PoissonElementKernel<MemorySpace::Device>;

} // namespace femx::examples::poisson
