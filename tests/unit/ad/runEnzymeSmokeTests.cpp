#include <cmath>
#include <iostream>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/EnzymeBoundaryKernel.hpp>
#include <femx/assembly/EnzymeVolumeKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace
{

femx::Real polynomial(femx::Real x)
{
  return x * x * x + femx::constants::TWO * x;
}

void localVolumeResidual(femx::Index       cell,
                         femx::Index       num_qp,
                         femx::Index       num_nodes,
                         femx::Index       dim,
                         femx::Index       num_res,
                         femx::Index       num_states,
                         femx::Index       num_params,
                         const femx::Real* N,
                         const femx::Real* dNdx,
                         const femx::Real* JxW,
                         const femx::Real* u,
                         const femx::Real* m,
                         femx::Real*       out)
{
  (void) cell;
  (void) dim;
  (void) num_res;
  (void) num_states;
  (void) num_params;
  (void) dNdx;

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

void localBoundaryResidual(femx::Index       facet,
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

  femx::Mesh           volume_mesh = femx::Mesh::makeStructuredQuad(1, 1);
  femx::LagrangeQuadQ1 volume_elem;
  femx::FESpace        volume_space(&volume_mesh, &volume_elem);
  volume_space.setup();

  femx::Vector<Real> u(4);
  u[0] = 1.5;
  u[1] = -0.25;
  u[2] = 0.75;
  u[3] = 2.0;

  femx::Vector<Real> m(4);
  m[0] = 0.2;
  m[1] = -0.4;
  m[2] = 0.6;
  m[3] = 0.8;

  const auto volume_quad = femx::GaussQuadrature::quadrilateral(1);
  femx::assembly::EnzymeVolumeKernel<localVolumeResidual>
      volume_kernel(volume_space, volume_quad, 4, 4, 4);

  femx::Vector<Real> volume_res;
  volume_kernel.res(0, u, m, volume_res);

  const femx::Real volume_uq =
      0.25 * (u[0] + u[1] + u[2] + u[3]);
  const femx::Real volume_mq =
      0.25 * (m[0] + m[1] + m[2] + m[3]);
  const femx::Real volume_source =
      volume_uq * volume_uq + 2.0 * volume_mq;

  for (femx::Index i = 0; i < 4; ++i)
  {
    if (!near(volume_res[i], 0.25 * volume_source))
    {
      std::cerr << "Incorrect Enzyme volume residual\n";
      return 1;
    }
  }

  femx::DenseMatrix volume_state_jac;
  volume_kernel.stateJac(0, u, m, volume_state_jac);

  for (femx::Index i = 0; i < 4; ++i)
  {
    for (femx::Index j = 0; j < 4; ++j)
    {
      if (!near(volume_state_jac(i, j), 0.125 * volume_uq))
      {
        std::cerr << "Incorrect Enzyme volume state Jacobian\n";
        return 1;
      }
    }
  }

  femx::DenseMatrix volume_param_jac;
  volume_kernel.paramJac(0, u, m, volume_param_jac);

  for (femx::Index i = 0; i < 4; ++i)
  {
    for (femx::Index j = 0; j < 4; ++j)
    {
      if (!near(volume_param_jac(i, j), 0.125))
      {
        std::cerr << "Incorrect Enzyme volume parameter Jacobian\n";
        return 1;
      }
    }
  }

  femx::Vector<Real> boundary_u(2);
  boundary_u[0] = 1.5;
  boundary_u[1] = -0.25;

  femx::Vector<Real> boundary_m(2);
  boundary_m[0] = 0.2;
  boundary_m[1] = -0.4;

  femx::Mesh mesh(2);
  mesh.addNode({0.0, 0.0, 0.0});
  mesh.addNode({2.0, 0.0, 0.0});

  femx::Mesh::BoundaryFacet integral_facet;
  integral_facet.shape    = femx::Cell::Shape::Segment;
  integral_facet.node_ids = {0, 1};

  const auto quad = femx::GaussQuadrature::segment(1);
  femx::assembly::EnzymeBoundaryKernel<localBoundaryResidual>
      boundary_kernel(mesh, quad, 2, 2, 2);

  femx::Vector<Real> integral_res;
  boundary_kernel.res(0, integral_facet, boundary_u, boundary_m, integral_res);

  const femx::Real uq     = 0.5 * (boundary_u[0] + boundary_u[1]);
  const femx::Real mq     = 0.5 * (boundary_m[0] + boundary_m[1]);
  const femx::Real source = uq * uq + 2.0 * mq;

  if (!near(integral_res[0], source) || !near(integral_res[1], source))
  {
    std::cerr << "Incorrect Enzyme boundary integral residual\n";
    return 1;
  }

  femx::DenseMatrix integral_state_jac;
  boundary_kernel.stateJac(
      0, integral_facet, boundary_u, boundary_m, integral_state_jac);

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
  boundary_kernel.paramJac(
      0, integral_facet, boundary_u, boundary_m, integral_param_jac);

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
