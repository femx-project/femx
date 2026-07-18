#pragma once

#include <femx/assembly/Assembly.hpp>

namespace femx::examples::poisson
{

/** @brief Q1 Laplace element operator shared by the CPU and CUDA examples. */
struct PoissonQuadQ1Operator
{
  /** @brief Evaluate one local residual row and its Jacobian row. */
  template <MemorySpace Space>
  FEMX_HOST_DEVICE void evalRow(
      const assembly::ElementView<Space>& e,
      Index                               row,
      Real&                               res,
      VectorView<Space, Real>             jac) const
  {
    constexpr Real gauss = 0.57735026918962576451;

    res = 0.0;
    for (Index col = 0; col < jac.size(); ++col)
    {
      jac[col] = 0.0;
    }

    for (Index qr = 0; qr < 2; ++qr)
    {
      const Real r = qr == 0 ? -gauss : gauss;
      for (Index qs = 0; qs < 2; ++qs)
      {
        const Real s = qs == 0 ? -gauss : gauss;
        Real       dNdr[4][2];
        refGrads(r, s, dNdr);

        Real J[4] = {0.0, 0.0, 0.0, 0.0};
        for (Index in = 0; in < 4; ++in)
        {
          const Real x  = e.coords[in * 2];
          const Real y  = e.coords[in * 2 + 1];
          J[0]         += x * dNdr[in][0];
          J[1]         += x * dNdr[in][1];
          J[2]         += y * dNdr[in][0];
          J[3]         += y * dNdr[in][1];
        }

        const Real det     = J[0] * J[3] - J[1] * J[2];
        const Real inv_det = 1.0 / det;
        const Real invJ[4] = {J[3] * inv_det,
                              -J[1] * inv_det,
                              -J[2] * inv_det,
                              J[0] * inv_det};
        const Real JxW     = det < 0.0 ? -det : det;

        Real grads[4][2];
        for (Index in = 0; in < 4; ++in)
        {
          grads[in][0] = dNdr[in][0] * invJ[0] + dNdr[in][1] * invJ[2];
          grads[in][1] = dNdr[in][0] * invJ[1] + dNdr[in][1] * invJ[3];
        }

        for (Index col = 0; col < jac.size(); ++col)
        {
          jac[col] +=
              (grads[row][0] * grads[col][0]
               + grads[row][1] * grads[col][1])
              * JxW;
        }
      }
    }

    for (Index col = 0; col < jac.size(); ++col)
    {
      res += jac[col] * e.state[col];
    }
  }

private:
  FEMX_HOST_DEVICE static void refGrads(Real r,
                                        Real s,
                                        Real (&out)[4][2])
  {
    out[0][0] = -0.25 * (1.0 - s);
    out[0][1] = -0.25 * (1.0 - r);
    out[1][0] = 0.25 * (1.0 - s);
    out[1][1] = -0.25 * (1.0 + r);
    out[2][0] = 0.25 * (1.0 + s);
    out[2][1] = 0.25 * (1.0 + r);
    out[3][0] = -0.25 * (1.0 + s);
    out[3][1] = 0.25 * (1.0 - r);
  }
};

} // namespace femx::examples::poisson
