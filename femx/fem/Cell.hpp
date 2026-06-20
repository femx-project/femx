#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <femx/core/Types.hpp>

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

  Cell(std::vector<Index> node_ids,
       std::vector<Node>  nodes,
       Shape              shape         = Shape::Unknown,
       Index              entity_dim    = 0,
       Index              entity_tag    = 0,
       Index              physical_tag  = 0,
       std::string        physical_name = {})
    : node_ids_(std::move(node_ids)),
      nodes_(std::move(nodes)),
      shape_(shape),
      entity_dim_(entity_dim),
      entity_tag_(entity_tag),
      physical_tag_(physical_tag),
      physical_name_(std::move(physical_name))
  {
  }

  Index numNodes() const
  {
    return static_cast<Index>(nodes_.size());
  }

  const Index* nodeIdsData() const
  {
    return node_ids_.data();
  }

  const std::vector<Index>& nodeIds() const
  {
    return node_ids_;
  }

  const Node& node(Index in) const
  {
    return nodes_[static_cast<std::size_t>(in)];
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
  std::vector<Index> node_ids_;
  std::vector<Node>  nodes_;
  Shape              shape_{Shape::Unknown};
  Index              entity_dim_{0};
  Index              entity_tag_{0};
  Index              physical_tag_{0};
  std::string        physical_name_;
};

} // namespace femx
