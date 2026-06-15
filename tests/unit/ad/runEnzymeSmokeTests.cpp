#include <cmath>
#include <iostream>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/EnzymeBoundaryElementKernel.hpp>
#include <femx/assembly/EnzymeBoundaryIntegralKernel.hpp>
#include <femx/assembly/EnzymeElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace
{

femx::Real polynomial(femx::Real x)
{
  return x * x * x + femx::constants::TWO * x;
}

void localResidual(femx::Index       ic,
                   femx::Index       num_res,
                   femx::Index       num_states,
                   femx::Index       num_params,
                   const femx::Real* u,
                   const femx::Real* m,
                   femx::Real*       out)
{
  (void) num_res;
  (void) num_states;
  (void) num_params;

  const femx::Real scale = 1.0 + static_cast<femx::Real>(ic);
  out[0]                 = scale * u[0] * u[0] + u[1] + 2.0 * m[0];
  out[1]                 = u[0] * u[1] + m[0] * m[0] + 3.0 * m[1];
}

void localBoundaryResidual(femx::Index       ib,
                           femx::Index       entity_tag,
                           femx::Index       physical_tag,
                           femx::Index       num_res,
                           femx::Index       num_states,
                           femx::Index       num_params,
                           const femx::Real* u,
                           const femx::Real* m,
                           femx::Real*       out)
{
  (void) ib;
  (void) num_res;
  (void) num_states;
  (void) num_params;

  const femx::Real state_scale =
      1.0 + static_cast<femx::Real>(entity_tag);
  const femx::Real param_scale =
      0.5 * static_cast<femx::Real>(physical_tag);

  out[0] = state_scale * u[0] * u[0] + u[1] + 2.0 * m[0];
  out[1] = u[0] * u[1] + param_scale * m[0] * m[0] + 3.0 * m[1];
}

void localBoundaryIntegralResidual(femx::Index       facet,
                                   femx::Index       num_qp,
                                   femx::Index       num_nodes,
                                   femx::Index       dim,
                                   femx::Index       num_res,
                                   femx::Index       num_states,
                                   femx::Index       num_params,
                                   const femx::Real* N,
                                   const femx::Real* point,
                                   const femx::Real* normal,
                                   const femx::Real* JxW,
                                   const femx::Real* u,
                                   const femx::Real* m,
                                   femx::Real*       out)
{
  (void) facet;
  (void) dim;
  (void) num_res;
  (void) num_states;
  (void) num_params;
  (void) point;
  (void) normal;

  for (femx::Index i = 0; i < num_nodes; ++i)
  {
    out[i] = 0.0;
  }

  for (femx::Index iq = 0; iq < num_qp; ++iq)
  {
    const femx::Real* Nq = N + iq * num_nodes;

    femx::Real uq = 0.0;
    femx::Real mq = 0.0;
    for (femx::Index a = 0; a < num_nodes; ++a)
    {
      uq += Nq[a] * u[a];
      mq += Nq[a] * m[a];
    }

    const femx::Real source = uq * uq + 2.0 * mq;
    for (femx::Index a = 0; a < num_nodes; ++a)
    {
      out[a] += Nq[a] * source * JxW[iq];
    }
  }
}

bool near(femx::Real actual, femx::Real expected)
{
  return std::abs(actual - expected) < 1.0e-10;
}

} // namespace

int main(int, char**)
{
  const femx::Real x        = 3.0;
  const femx::Real expected = 3.0 * x * x + femx::constants::TWO;
  const femx::Real actual   = femx::ad::derivative<polynomial>(x);

  if (std::abs(actual - expected) > 1.0e-10)
  {
    std::cerr << "Incorrect Enzyme derivative: expected " << expected
              << ", got " << actual << '\n';
    return 1;
  }

  femx::assembly::EnzymeElementKernel<localResidual> kernel(2, 2, 2);

  femx::Vector u(2);
  u[0] = 1.5;
  u[1] = -0.25;

  femx::Vector m(2);
  m[0] = 0.2;
  m[1] = -0.4;

  femx::Vector res;
  kernel.res(2, u, m, res);

  const femx::Real scale = 3.0;
  if (!near(res[0], scale * u[0] * u[0] + u[1] + 2.0 * m[0])
      || !near(res[1], u[0] * u[1] + m[0] * m[0] + 3.0 * m[1]))
  {
    std::cerr << "Incorrect Enzyme residual evaluation\n";
    return 1;
  }

  femx::DenseMatrix state_jac;
  kernel.stateJac(2, u, m, state_jac);

  if (!near(state_jac(0, 0), 2.0 * scale * u[0])
      || !near(state_jac(0, 1), 1.0)
      || !near(state_jac(1, 0), u[1])
      || !near(state_jac(1, 1), u[0]))
  {
    std::cerr << "Incorrect Enzyme state Jacobian\n";
    return 1;
  }

  femx::DenseMatrix param_jac;
  kernel.paramJac(2, u, m, param_jac);

  if (!near(param_jac(0, 0), 2.0)
      || !near(param_jac(0, 1), 0.0)
      || !near(param_jac(1, 0), 2.0 * m[0])
      || !near(param_jac(1, 1), 3.0))
  {
    std::cerr << "Incorrect Enzyme parameter Jacobian\n";
    return 1;
  }

  femx::Mesh::BoundaryFacet facet;
  facet.entity_tag   = 4;
  facet.physical_tag = 6;

  femx::assembly::EnzymeBoundaryElementKernel<localBoundaryResidual>
      boundary_kernel(2, 2, 2);

  femx::Vector boundary_res;
  boundary_kernel.res(3, facet, u, m, boundary_res);

  const femx::Real state_scale = 5.0;
  const femx::Real param_scale = 3.0;
  if (!near(boundary_res[0], state_scale * u[0] * u[0] + u[1] + 2.0 * m[0])
      || !near(boundary_res[1],
               u[0] * u[1] + param_scale * m[0] * m[0] + 3.0 * m[1]))
  {
    std::cerr << "Incorrect Enzyme boundary residual evaluation\n";
    return 1;
  }

  femx::DenseMatrix boundary_state_jac;
  boundary_kernel.stateJac(3, facet, u, m, boundary_state_jac);

  if (!near(boundary_state_jac(0, 0), 2.0 * state_scale * u[0])
      || !near(boundary_state_jac(0, 1), 1.0)
      || !near(boundary_state_jac(1, 0), u[1])
      || !near(boundary_state_jac(1, 1), u[0]))
  {
    std::cerr << "Incorrect Enzyme boundary state Jacobian\n";
    return 1;
  }

  femx::DenseMatrix boundary_param_jac;
  boundary_kernel.paramJac(3, facet, u, m, boundary_param_jac);

  if (!near(boundary_param_jac(0, 0), 2.0)
      || !near(boundary_param_jac(0, 1), 0.0)
      || !near(boundary_param_jac(1, 0), 2.0 * param_scale * m[0])
      || !near(boundary_param_jac(1, 1), 3.0))
  {
    std::cerr << "Incorrect Enzyme boundary parameter Jacobian\n";
    return 1;
  }

  femx::Mesh mesh(2);
  mesh.addNode({0.0, 0.0, 0.0});
  mesh.addNode({2.0, 0.0, 0.0});

  femx::Mesh::BoundaryFacet integral_facet;
  integral_facet.shape    = femx::Cell::Shape::Segment;
  integral_facet.node_ids = {0, 1};

  const auto quad = femx::GaussQuadrature::segment(1);
  femx::assembly::EnzymeBoundaryIntegralKernel<
      localBoundaryIntegralResidual>
      integral_kernel(mesh, quad, 2, 2, 2);

  femx::Vector integral_res;
  integral_kernel.res(0, integral_facet, u, m, integral_res);

  const femx::Real uq     = 0.5 * (u[0] + u[1]);
  const femx::Real mq     = 0.5 * (m[0] + m[1]);
  const femx::Real source = uq * uq + 2.0 * mq;

  if (!near(integral_res[0], source) || !near(integral_res[1], source))
  {
    std::cerr << "Incorrect Enzyme boundary integral residual\n";
    return 1;
  }

  femx::DenseMatrix integral_state_jac;
  integral_kernel.stateJac(0, integral_facet, u, m, integral_state_jac);

  for (femx::Index i = 0; i < 2; ++i)
  {
    for (femx::Index j = 0; j < 2; ++j)
    {
      if (!near(integral_state_jac(i, j), uq))
      {
        std::cerr << "Incorrect Enzyme boundary integral state Jacobian\n";
        return 1;
      }
    }
  }

  femx::DenseMatrix integral_param_jac;
  integral_kernel.paramJac(0, integral_facet, u, m, integral_param_jac);

  for (femx::Index i = 0; i < 2; ++i)
  {
    for (femx::Index j = 0; j < 2; ++j)
    {
      if (!near(integral_param_jac(i, j), 1.0))
      {
        std::cerr << "Incorrect Enzyme boundary integral parameter Jacobian\n";
        return 1;
      }
    }
  }

  return 0;
}
