#pragma once

#include <array>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class Cell
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

  Cell() = default;

  Cell(Vector<Index> nids,
       Vector<Node>  nodes,
       Shape         shape      = Shape::Unknown,
       Index         edim       = 0,
       Index         entity_tag = 0,
       Index         ptag       = 0,
       std::string   pname      = {})
    : node_ids_(std::move(nids)),
      nodes_(std::move(nodes)),
      shape_(shape),
      entity_dim_(edim),
      entity_tag_(entity_tag),
      physical_tag_(ptag),
      physical_name_(std::move(pname))
  {
  }

  Index numNodes() const
  {
    return nodes_.size();
  }

  const Index* nodeIdsData() const
  {
    return node_ids_.data();
  }

  const Vector<Index>& nodeIds() const
  {
    return node_ids_;
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
    return entity_dim_;
  }

  Index entityTag() const noexcept
  {
    return entity_tag_;
  }

  Index physicalTag() const noexcept
  {
    return physical_tag_;
  }

  const std::string& physicalName() const noexcept
  {
    return physical_name_;
  }

private:
  Vector<Index> node_ids_;
  Vector<Node>  nodes_;
  Shape         shape_{Shape::Unknown};
  Index         entity_dim_{0};
  Index         entity_tag_{0};
  Index         physical_tag_{0};
  std::string   physical_name_;
};

} // namespace femx
