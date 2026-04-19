#pragma once

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include <refem/common/Types.hpp>
#include <refem/fe/ReferenceElement.hpp>

namespace refem
{

struct QuadraturePoint
{
  real_type x[3]   = {0.0, 0.0, 0.0};
  real_type weight = 0.0;

  real_type operator[](index_type i) const
  {
    return x[i];
  }
};

class GaussQuadrature
{
public:
  GaussQuadrature() = default;

  GaussQuadrature(ReferenceElement             elem,
                  index_type                   dim,
                  std::vector<QuadraturePoint> points)
    : cell_(elem),
      dim_(dim),
      points_(std::move(points))
  {
  }

  index_type size() const
  {
    return static_cast<index_type>(points_.size());
  }

  index_type dim() const
  {
    return dim_;
  }

  ReferenceElement referenceElement() const
  {
    return cell_;
  }

  const QuadraturePoint& operator[](index_type q) const
  {
    return points_[q];
  }

  static GaussQuadrature make(ReferenceElement cell, index_type order)
  {
    switch (cell)
    {
    case ReferenceElement::Segment:
      return segment(order);

    case ReferenceElement::Triangle:
      return triangle(order);

    case ReferenceElement::Quadrilateral:
      return quadrilateral(order);
    }

    throw std::runtime_error("Unsupported reference cell");
  }

  static GaussQuadrature segment(index_type n)
  {
    switch (n)
    {
    case 1:
    {
      return GaussQuadrature(
          ReferenceElement::Segment,
          1,
          {
              QuadraturePoint{{0.0, 0.0, 0.0}, 2.0},
          });
    }

    case 2:
    {
      return GaussQuadrature(
          ReferenceElement::Segment,
          1,
          {
              QuadraturePoint{{-0.5773502691896257, 0.0, 0.0}, 1.0},
              QuadraturePoint{{0.5773502691896257, 0.0, 0.0}, 1.0},
          });
    }

    case 3:
    {
      return GaussQuadrature(
          ReferenceElement::Segment,
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

  static GaussQuadrature quadrilateral(index_type n)
  {
    switch (n)
    {
    case 1:
    {
      return GaussQuadrature(
          ReferenceElement::Quadrilateral,
          2,
          {
              QuadraturePoint{{0.0, 0.0, 0.0}, 4.0},
          });
    }

    case 2:
    {
      return GaussQuadrature(
          ReferenceElement::Quadrilateral,
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
          ReferenceElement::Quadrilateral,
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

  static GaussQuadrature triangle(index_type order)
  {
    switch (order)
    {
    case 1:
    {
      return GaussQuadrature(
          ReferenceElement::Triangle,
          2,
          {
              QuadraturePoint{{1.0 / 3.0, 1.0 / 3.0, 0.0}, 0.5},
          });
    }

    case 2:
    {
      return GaussQuadrature(
          ReferenceElement::Triangle,
          2,
          {
              QuadraturePoint{{1.0 / 6.0, 1.0 / 6.0, 0.0}, 1.0 / 6.0},
              QuadraturePoint{{2.0 / 3.0, 1.0 / 6.0, 0.0}, 1.0 / 6.0},
              QuadraturePoint{{1.0 / 6.0, 2.0 / 3.0, 0.0}, 1.0 / 6.0},
          });
    }

    default:
      throw std::runtime_error("Unsupported triangle quadrature order");
    }
  }

private:
  ReferenceElement             cell_ = ReferenceElement::Segment;
  index_type                   dim_  = 0;
  std::vector<QuadraturePoint> points_;
};

} // namespace refem
