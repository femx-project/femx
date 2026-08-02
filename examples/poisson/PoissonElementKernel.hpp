#pragma once

#include <femx/assembly/Assembly.hpp>
#include <femx/fem/ElementQuadData.hpp>

namespace femx::examples::poisson
{

/**
 * @brief Evaluate the scalar Laplace operator from element quadrature data.
 *
 * For a local row `i`, this kernel forms `K_ij^e = sum_q grad(N_i) dot grad(N_j) JxW_q`
 * and then evaluates the element residual `R_i^e = sum_j K_ij^e x_j^e`.
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
    for (Index j = 0; j < jac.size(); ++j)
    {
      jac[j] = 0.0;
    }

    // Integrate row `row` of the element stiffness matrix K^e.
    for (Index iq = 0; iq < data_.numQuadraturePoints(); ++iq)
    {
      for (Index j = 0; j < jac.size(); ++j)
      {
        Real grad_dot = 0.0;
        for (Index id = 0; id < data_.dim(); ++id)
        {
          grad_dot += data_.dNdx(elem.ie, iq, row, id) * data_.dNdx(elem.ie, iq, j, id);
        }
        jac[j] += grad_dot * data_.JxW(elem.ie, iq);
      }
    }

    // Apply that row to the element state to obtain R_row^e.
    for (Index j = 0; j < jac.size(); ++j)
    {
      res += jac[j] * elem.state[j];
    }
  }

private:
  fem::ElementQuadDataView<Space> data_; ///< Element integration data.
};

using HostPoissonElementKernel   = PoissonElementKernel<MemorySpace::Host>;
using DevicePoissonElementKernel = PoissonElementKernel<MemorySpace::Device>;

} // namespace femx::examples::poisson
