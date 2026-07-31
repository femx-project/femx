#pragma once

#include <array>
#include <stdexcept>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{
namespace fem
{

struct QuadraturePoint
{
  Real x[3] = {0.0, 0.0, 0.0}; ///< Reference coordinates.
  Real wt   = 0.0;             ///< Quadrature weight.

  Real operator[](Index i) const
  {
    return x[i];
  }
};

class GaussQuadrature
{
public:
  GaussQuadrature() = default;

  GaussQuadrature(ElementShape                shape,
                  Index                       dim,
                  HostVector<QuadraturePoint> pts)
    : shape_(shape),
      dim_(dim),
      pts_(std::move(pts))
  {
  }

  Index size() const
  {
    return pts_.size();
  }

  Index dim() const
  {
    return dim_;
  }

  ElementShape shape() const
  {
    return shape_;
  }

  const QuadraturePoint& operator[](Index iq) const
  {
    return pts_[iq];
  }

  static GaussQuadrature make(ElementShape shape, Index order)
  {
    switch (shape)
    {
    case ElementShape::Segment:
      return segment(order);

    case ElementShape::Triangle:
      return triangle(order);

    case ElementShape::Quadrilateral:
      return quadrilateral(order);

    case ElementShape::Tetrahedron:
      return tetrahedron(order);

    case ElementShape::Unknown:
    case ElementShape::Hexahedron:
      break;
    }

    throw std::runtime_error("Unsupported reference elem");
  }

  static GaussQuadrature segment(Index num_points)
  {
    switch (num_points)
    {
    case 1:
    {
      return GaussQuadrature(
          ElementShape::Segment,
          1,
          {
              QuadraturePoint{{0.0, 0.0, 0.0}, 2.0},
          });
    }

    case 2:
    {
      return GaussQuadrature(
          ElementShape::Segment,
          1,
          {
              QuadraturePoint{{-0.5773502691896257, 0.0, 0.0}, 1.0},
              QuadraturePoint{{0.5773502691896257, 0.0, 0.0}, 1.0},
          });
    }

    case 3:
    {
      return GaussQuadrature(
          ElementShape::Segment,
          1,
          {
              QuadraturePoint{{-0.7745966692414834, 0.0, 0.0}, 5.0 / 9.0},
              QuadraturePoint{{0.0, 0.0, 0.0}, 8.0 / 9.0},
              QuadraturePoint{{0.7745966692414834, 0.0, 0.0}, 5.0 / 9.0},
          });
    }

    default:
      throw std::runtime_error("Unsupported number of Gauss points for segment");
    }
  }

  static GaussQuadrature quadrilateral(Index num_points)
  {
    switch (num_points)
    {
    case 1:
    {
      return GaussQuadrature(
          ElementShape::Quadrilateral,
          2,
          {
              QuadraturePoint{{0.0, 0.0, 0.0}, 4.0},
          });
    }

    case 2:
    {
      return GaussQuadrature(
          ElementShape::Quadrilateral,
          2,
          {
              QuadraturePoint{{-0.5773502691896257, -0.5773502691896257, 0.0}, 1.0},
              QuadraturePoint{{0.5773502691896257, -0.5773502691896257, 0.0}, 1.0},
              QuadraturePoint{{0.5773502691896257, 0.5773502691896257, 0.0}, 1.0},
              QuadraturePoint{{-0.5773502691896257, 0.5773502691896257, 0.0}, 1.0},
          });
    }

    case 3:
    {
      return GaussQuadrature(
          ElementShape::Quadrilateral,
          2,
          {
              QuadraturePoint{{-0.7745966692414834, -0.7745966692414834, 0.0}, 25.0 / 81.0},
              QuadraturePoint{{0.0, -0.7745966692414834, 0.0}, 40.0 / 81.0},
              QuadraturePoint{{0.7745966692414834, -0.7745966692414834, 0.0}, 25.0 / 81.0},

              QuadraturePoint{{-0.7745966692414834, 0.0, 0.0}, 40.0 / 81.0},
              QuadraturePoint{{0.0, 0.0, 0.0}, 64.0 / 81.0},
              QuadraturePoint{{0.7745966692414834, 0.0, 0.0}, 40.0 / 81.0},

              QuadraturePoint{{-0.7745966692414834, 0.7745966692414834, 0.0}, 25.0 / 81.0},
              QuadraturePoint{{0.0, 0.7745966692414834, 0.0}, 40.0 / 81.0},
              QuadraturePoint{{0.7745966692414834, 0.7745966692414834, 0.0}, 25.0 / 81.0},
          });
    }

    default:
      throw std::runtime_error("Unsupported number of Gauss points for quadrilateral");
    }
  }

  static GaussQuadrature triangle(Index order)
  {
    switch (order)
    {
    case 1:
    {
      return GaussQuadrature(
          ElementShape::Triangle,
          2,
          {
              QuadraturePoint{{1.0 / 3.0, 1.0 / 3.0, 0.0}, 0.5},
          });
    }

    case 2:
    {
      return GaussQuadrature(
          ElementShape::Triangle,
          2,
          {
              QuadraturePoint{{1.0 / 6.0, 1.0 / 6.0, 0.0}, 1.0 / 6.0},
              QuadraturePoint{{2.0 / 3.0, 1.0 / 6.0, 0.0}, 1.0 / 6.0},
              QuadraturePoint{{1.0 / 6.0, 2.0 / 3.0, 0.0}, 1.0 / 6.0},
          });
    }

    default:
      throw std::runtime_error("Unsupported triangle quad order");
    }
  }

  static GaussQuadrature tetrahedron(Index order)
  {
    switch (order)
    {
    case 1:
    {
      return GaussQuadrature(
          ElementShape::Tetrahedron,
          3,
          {
              QuadraturePoint{{0.25, 0.25, 0.25}, 1.0 / 6.0},
          });
    }

    case 2:
    {
      constexpr Real a = 0.1381966011250105;
      constexpr Real b = 0.5854101966249685;
      return GaussQuadrature(
          ElementShape::Tetrahedron,
          3,
          {
              QuadraturePoint{{a, a, a}, 1.0 / 24.0},
              QuadraturePoint{{b, a, a}, 1.0 / 24.0},
              QuadraturePoint{{a, b, a}, 1.0 / 24.0},
              QuadraturePoint{{a, a, b}, 1.0 / 24.0},
          });
    }

    default:
      throw std::runtime_error("Unsupported tetrahedron quad order");
    }
  }

private:
  ElementShape                shape_ = ElementShape::Segment;
  Index                       dim_   = 0;
  HostVector<QuadraturePoint> pts_;
};

} // namespace fem
} // namespace femx
