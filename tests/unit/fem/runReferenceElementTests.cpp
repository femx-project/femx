#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "TestHelper.hpp"
#include <femx/fem/ElementQuadratureData.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

bool near(Real a, Real b)
{
  return std::abs(a - b) <= 1.0e-12;
}

template <std::size_t N>
bool valsNear(const HostVector&          actual,
              const std::array<Real, N>& expected)
{
  if (actual.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[static_cast<Index>(i)], expected[i]))
    {
      return false;
    }
  }
  return true;
}

HostVector shapeVals(const FiniteElement&   elem,
                     const QuadraturePoint& qp)
{
  HostVector vals(elem.numDofsPerElement());
  elem.calcN(qp, HostVectorView(vals.data(), vals.size()));
  return vals;
}

DenseMatrix shapeGradients(const FiniteElement&   elem,
                           const QuadraturePoint& qp)
{
  DenseMatrix gradients(elem.numDofsPerElement(), elem.dim());
  elem.calcdNdr(qp,
                HostMatrixView<Real>(
                    gradients.data(), gradients.rows(), gradients.cols()));
  return gradients;
}

Real sum(const HostVector& vals)
{
  Real total = 0.0;
  for (Index i = 0; i < vals.size(); ++i)
  {
    total += vals[i];
  }
  return total;
}

bool gradientColumnsSumToZero(const FiniteElement&   elem,
                              const QuadraturePoint& qp)
{
  const DenseMatrix gradients = shapeGradients(elem, qp);
  for (Index dim = 0; dim < elem.dim(); ++dim)
  {
    Real total = 0.0;
    for (Index node = 0; node < elem.numDofsPerElement(); ++node)
    {
      total += gradients(node, dim);
    }
    if (!near(total, 0.0))
    {
      return false;
    }
  }
  return true;
}

Real quadratureWeightSum(const GaussQuadrature& quad)
{
  Real total = 0.0;
  for (Index iq = 0; iq < quad.size(); ++iq)
  {
    total += quad[iq].wt;
  }
  return total;
}

TestOutcome elementQuadratureDataMapsToPhysicalSpace()
{
  TestStatus status(__func__);

  Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
  LagrangeQuadQ1 quad_element;
  FESpace        space(&mesh, &quad_element);
  space.setup();

  const auto data = makeElementQuadratureData(
      space,
      GaussQuadrature::make(ReferenceElement::Quadrilateral, 2));
  const auto view = data.view();

  status *= data.numElems() == 1;
  status *= data.numQuadraturePoints() == 4;
  status *= data.numShapes() == 4;
  status *= data.dim() == 2;

  Real volume = 0.0;
  for (Index iq = 0; iq < data.numQuadraturePoints(); ++iq)
  {
    Real shape_sum = 0.0;
    Real grad_sum[2]{0.0, 0.0};
    for (Index shape = 0; shape < data.numShapes(); ++shape)
    {
      shape_sum += view.N(iq, shape);
      for (Index d = 0; d < data.dim(); ++d)
      {
        grad_sum[d] += view.dNdx(0, iq, shape, d);
      }
    }
    status *= near(shape_sum, 1.0);
    status *= near(grad_sum[0], 0.0);
    status *= near(grad_sum[1], 0.0);
    status *= near(view.JxW(0, iq), 0.25);
    volume += view.JxW(0, iq);
  }
  status *= near(volume, 1.0);

  return status.report();
}

TestOutcome elementMetadata()
{
  TestStatus status(__func__);

  LagrangeTriangleP1 tri;
  status *= tri.name() == "LagrangeTriangleP1";
  status *= tri.dim() == 2;
  status *= tri.numNodes() == 3;
  status *= tri.numDofsPerElement() == 3;
  status *= tri.order() == 1;
  status *= tri.referenceElement() == ReferenceElement::Triangle;

  LagrangeQuadQ1 quad;
  status *= quad.name() == "LagrangeQuadQ1";
  status *= quad.dim() == 2;
  status *= quad.numNodes() == 4;
  status *= quad.numDofsPerElement() == 4;
  status *= quad.order() == 1;
  status *= quad.referenceElement() == ReferenceElement::Quadrilateral;

  LagrangeTetrahedronP1 tet;
  status *= tet.name() == "LagrangeTetrahedronP1";
  status *= tet.dim() == 3;
  status *= tet.numNodes() == 4;
  status *= tet.numDofsPerElement() == 4;
  status *= tet.order() == 1;
  status *= tet.referenceElement() == ReferenceElement::Tetrahedron;

  return status.report();
}

TestOutcome triangleP1ShapeFunctions()
{
  TestStatus status(__func__);

  LagrangeTriangleP1    tri;
  const QuadraturePoint interior{{0.2, 0.3, 0.0}, 0.0};
  const HostVector      N = shapeVals(tri, interior);

  status *= valsNear(N, std::array<Real, 3>{{0.5, 0.2, 0.3}});
  status *= near(sum(N), 1.0);
  status *= gradientColumnsSumToZero(tri, interior);

  const std::array<QuadraturePoint, 3> nodes{{
      QuadraturePoint{{0.0, 0.0, 0.0}, 0.0},
      QuadraturePoint{{1.0, 0.0, 0.0}, 0.0},
      QuadraturePoint{{0.0, 1.0, 0.0}, 0.0},
  }};
  for (Index node = 0; node < tri.numNodes(); ++node)
  {
    const HostVector vals = shapeVals(
        tri, nodes[static_cast<std::size_t>(node)]);
    for (Index i = 0; i < tri.numNodes(); ++i)
    {
      status *= near(vals[i], i == node ? 1.0 : 0.0);
    }
  }

  return status.report();
}

TestOutcome quadQ1ShapeFunctions()
{
  TestStatus status(__func__);

  LagrangeQuadQ1        quad;
  const QuadraturePoint interior{{0.2, -0.4, 0.0}, 0.0};
  const HostVector      N = shapeVals(quad, interior);

  status *= valsNear(N, std::array<Real, 4>{{0.28, 0.42, 0.18, 0.12}});
  status *= near(sum(N), 1.0);
  status *= gradientColumnsSumToZero(quad, interior);

  const std::array<QuadraturePoint, 4> nodes{{
      QuadraturePoint{{-1.0, -1.0, 0.0}, 0.0},
      QuadraturePoint{{1.0, -1.0, 0.0}, 0.0},
      QuadraturePoint{{1.0, 1.0, 0.0}, 0.0},
      QuadraturePoint{{-1.0, 1.0, 0.0}, 0.0},
  }};
  for (Index node = 0; node < quad.numNodes(); ++node)
  {
    const HostVector vals = shapeVals(
        quad, nodes[static_cast<std::size_t>(node)]);
    for (Index i = 0; i < quad.numNodes(); ++i)
    {
      status *= near(vals[i], i == node ? 1.0 : 0.0);
    }
  }

  return status.report();
}

TestOutcome tetrahedronP1ShapeFunctions()
{
  TestStatus status(__func__);

  LagrangeTetrahedronP1 tet;
  const QuadraturePoint interior{{0.2, 0.3, 0.1}, 0.0};
  const HostVector      N = shapeVals(tet, interior);

  status *= valsNear(N, std::array<Real, 4>{{0.4, 0.2, 0.3, 0.1}});
  status *= near(sum(N), 1.0);
  status *= gradientColumnsSumToZero(tet, interior);

  const std::array<QuadraturePoint, 4> nodes{{
      QuadraturePoint{{0.0, 0.0, 0.0}, 0.0},
      QuadraturePoint{{1.0, 0.0, 0.0}, 0.0},
      QuadraturePoint{{0.0, 1.0, 0.0}, 0.0},
      QuadraturePoint{{0.0, 0.0, 1.0}, 0.0},
  }};
  for (Index node = 0; node < tet.numNodes(); ++node)
  {
    const HostVector vals = shapeVals(
        tet, nodes[static_cast<std::size_t>(node)]);
    for (Index i = 0; i < tet.numNodes(); ++i)
    {
      status *= near(vals[i], i == node ? 1.0 : 0.0);
    }
  }

  return status.report();
}

TestOutcome quadratureIntegratesConstants()
{
  TestStatus status(__func__);

  const GaussQuadrature segment = GaussQuadrature::make(ReferenceElement::Segment, 3);

  status *= segment.referenceElement() == ReferenceElement::Segment;
  status *= segment.dim() == 1;
  status *= segment.size() == 3;
  status *= near(quadratureWeightSum(segment), 2.0);

  const GaussQuadrature triangle = GaussQuadrature::make(ReferenceElement::Triangle, 2);

  status *= triangle.referenceElement() == ReferenceElement::Triangle;
  status *= triangle.dim() == 2;
  status *= triangle.size() == 3;
  status *= near(quadratureWeightSum(triangle), 0.5);

  const GaussQuadrature quad = GaussQuadrature::make(ReferenceElement::Quadrilateral, 2);

  status *= quad.referenceElement() == ReferenceElement::Quadrilateral;
  status *= quad.dim() == 2;
  status *= quad.size() == 4;
  status *= near(quadratureWeightSum(quad), 4.0);

  const GaussQuadrature tet  = GaussQuadrature::make(ReferenceElement::Tetrahedron, 2);
  status                    *= tet.referenceElement() == ReferenceElement::Tetrahedron;
  status                    *= tet.dim() == 3;
  status                    *= tet.size() == 4;
  status                    *= near(quadratureWeightSum(tet), 1.0 / 6.0);

  bool threw = false;
  try
  {
    GaussQuadrature::make(ReferenceElement::Triangle, 3);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::elementMetadata();
  results += femx::tests::triangleP1ShapeFunctions();
  results += femx::tests::quadQ1ShapeFunctions();
  results += femx::tests::tetrahedronP1ShapeFunctions();
  results += femx::tests::quadratureIntegratesConstants();
  results += femx::tests::elementQuadratureDataMapsToPhysicalSpace();

  return results.summary();
}
