#pragma once

#include <array>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

class Element
{
public:
  using Node = std::array<Real, 3>;

  enum class Shape
  {
    Unknown,
    Segment,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron
  };

  Element() = default;

  Element(Vector<Index> nids,
          Vector<Node>  nodes,
          Shape         shape = Shape::Unknown,
          Index         edim  = 0,
          Index         etag  = 0,
          Index         ptag  = 0,
          std::string   pname = {})
    : nids_(std::move(nids)),
      nodes_(std::move(nodes)),
      shape_(shape),
      edim_(edim),
      etag_(etag),
      ptag_(ptag),
      pname_(std::move(pname))
  {
  }

  Index numNodes() const
  {
    return nodes_.size();
  }

  const Index* nodeIdsData() const
  {
    return nids_.data();
  }

  const Vector<Index>& nodeIds() const
  {
    return nids_;
  }

  const Node& node(Index in) const
  {
    return nodes_[in];
  }

  Shape shape() const noexcept
  {
    return shape_;
  }

  Index entityDim() const noexcept
  {
    return edim_;
  }

  Index entityTag() const noexcept
  {
    return etag_;
  }

  Index physicalTag() const noexcept
  {
    return ptag_;
  }

  const std::string& physicalName() const noexcept
  {
    return pname_;
  }

private:
  Vector<Index> nids_;
  Vector<Node>  nodes_;
  Shape         shape_{Shape::Unknown};
  Index         edim_{0};
  Index         etag_{0};
  Index         ptag_{0};
  std::string   pname_;
};

} // namespace fem
} // namespace femx
