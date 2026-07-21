#pragma once

#include <femx/assembly/Assembly.hpp>
#include <femx/fem/ElementQuadratureData.hpp>

namespace femx::examples::poisson
{

/**
 * @brief Evaluate the scalar Laplace operator from element quadrature data.
 */
template <MemorySpace Space>
class PoissonComponents
{
public:
  /**
   * @brief Bind the element quadrature data used during assembly.
   *
   * @param[in] data - Shape gradients and weighted Jacobians.
   */
  FEMX_HOST_DEVICE explicit PoissonComponents(
      fem::ElementQuadratureDataView<Space> data)
    : data_(data)
  {
  }

  /**
   * @brief Evaluate one local residual row and its Jacobian row.
   *
   * @param[in] e - Element state and metadata.
   * @param[in] row - Local residual row.
   * @param[out] res - Local residual value.
   * @param[out] jac - Local Jacobian row.
   */
  FEMX_HOST_DEVICE void evalRow(
      const assembly::ElementView<Space>& e,
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
        for (Index d = 0; d < data_.dim(); ++d)
        {
          grad_dot += data_.dNdx(e.ie, iq, row, d) * data_.dNdx(e.ie, iq, col, d);
        }
        jac[col] += grad_dot * data_.JxW(e.ie, iq);
      }
    }

    for (Index col = 0; col < jac.size(); ++col)
    {
      res += jac[col] * e.state[col];
    }
  }

private:
  fem::ElementQuadratureDataView<Space> data_; ///< Element integration data.
};

} // namespace femx::examples::poisson
