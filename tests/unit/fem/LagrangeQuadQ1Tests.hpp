#pragma once

#include <array>
#include <iostream>
#include <string>

#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LagrangeQuadQ1Tests : public TestBase
{
public:
  TestOutcome elementMetadata()
  {
    TestStatus status;
    status = true;

    LagrangeQuadQ1 elem;

    status *= (elem.name() == "LagrangeQuadQ1");
    status *= (elem.dim() == LagrangeQuadQ1::spatial_dim);
    status *= (elem.numNodes() == LagrangeQuadQ1::nnodes);
    status *= (elem.numDofsPerElement() == LagrangeQuadQ1::ndofs);
    status *= (elem.order() == LagrangeQuadQ1::degree);
    status *= (elem.referenceElement() == ReferenceElement::Quadrilateral);

    return status.report(__func__);
  }

  TestOutcome shapeAtReferenceNodes()
  {
    TestStatus status;
    status = true;

    LagrangeQuadQ1 elem;

    const std::array<QuadraturePoint, LagrangeQuadQ1::nnodes> nodes = {
        QuadraturePoint{{-1.0, -1.0, 0.0}, 0.0},
        QuadraturePoint{{1.0, -1.0, 0.0}, 0.0},
        QuadraturePoint{{1.0, 1.0, 0.0}, 0.0},
        QuadraturePoint{{-1.0, 1.0, 0.0}, 0.0},
    };

    for (Index in = 0; in < LagrangeQuadQ1::nnodes; ++in)
    {
      std::array<Real, LagrangeQuadQ1::nnodes> values = {};
      elem.calcShape(nodes[in], VectorView<Real>(values.data(), values.size()));

      for (Index i = 0; i < LagrangeQuadQ1::nnodes; ++i)
      {
        const Real expected = (i == in) ? 1.0 : 0.0;
        if (!isEqual(values[i], expected))
        {
          std::cout << "Incorrect shape value N[" << i << "] at reference node " << in
                    << ": expected " << expected << ", got " << values[i] << "\n";
          status = false;
        }
      }
    }

    return status.report(__func__);
  }

  TestOutcome shapePartitionOfUnity()
  {
    TestStatus status;
    status = true;

    LagrangeQuadQ1 elem;

    const std::array<QuadraturePoint, 3> points = {
        QuadraturePoint{{0.0, 0.0, 0.0}, 0.0},
        QuadraturePoint{{0.5, -0.5, 0.0}, 0.0},
        QuadraturePoint{{-0.5, 0.5, 0.0}, 0.0},
    };

    for (const auto& point : points)
    {
      std::array<Real, LagrangeQuadQ1::nnodes> values = {};
      elem.calcShape(point, VectorView<Real>(values.data(), values.size()));

      Real sum = 0.0;
      for (const Real value : values)
      {
        sum += value;
      }

      if (!isEqual(sum, 1.0))
      {
        std::cout << "Shape functions do not sum to 1.0 at (" << point[0] << ", " << point[1]
                  << "): got " << sum << "\n";
        status = false;
      }
    }

    return status.report(__func__);
  }

  TestOutcome shapeGradientAtCenter()
  {
    TestStatus status;
    status = true;

    LagrangeQuadQ1 elem;

    const QuadraturePoint                                                  center{{0.0, 0.0, 0.0}, 0.0};
    std::array<Real, LagrangeQuadQ1::nnodes * LagrangeQuadQ1::spatial_dim> gradients = {};

    elem.calcShapeGrad(
        center,
        MatrixView<Real>(gradients.data(), LagrangeQuadQ1::nnodes, LagrangeQuadQ1::spatial_dim));

    const MatrixView<Real> dNdr(
        gradients.data(), LagrangeQuadQ1::nnodes, LagrangeQuadQ1::spatial_dim);

    const Real expected[LagrangeQuadQ1::nnodes][LagrangeQuadQ1::spatial_dim] = {
        {-0.25, -0.25},
        {0.25, -0.25},
        {0.25, 0.25},
        {-0.25, 0.25},
    };

    for (Index i = 0; i < LagrangeQuadQ1::nnodes; ++i)
    {
      for (Index d = 0; d < LagrangeQuadQ1::spatial_dim; ++d)
      {
        if (!isEqual(dNdr(i, d), expected[i][d]))
        {
          std::cout << "Incorrect gradient entry (" << i << ", " << d << "): expected "
                    << expected[i][d] << ", got " << dNdr(i, d) << "\n";
          status = false;
        }
      }
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx
