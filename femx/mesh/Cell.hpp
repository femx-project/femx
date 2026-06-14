#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

class Cell
{
public:
  using Node = std::array<real_type, 3>;

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

  Cell(std::vector<index_type> node_ids,
       std::vector<Node>       nodes,
       Shape                   shape         = Shape::Unknown,
       index_type              entity_dim    = 0,
       index_type              entity_tag    = 0,
       index_type              physical_tag  = 0,
       std::string             physical_name = {})
    : node_ids_(std::move(node_ids)),
      nodes_(std::move(nodes)),
      shape_(shape),
      entity_dim_(entity_dim),
      entity_tag_(entity_tag),
      physical_tag_(physical_tag),
      physical_name_(std::move(physical_name))
  {
  }

  index_type numNodes() const
  {
    return static_cast<index_type>(nodes_.size());
  }

  const index_type* nodeIdsData() const
  {
    return node_ids_.data();
  }

  const std::vector<index_type>& nodeIds() const
  {
    return node_ids_;
  }

  const Node& node(index_type i) const
  {
    return nodes_[static_cast<std::size_t>(i)];
  }

  Shape shape() const noexcept
  {
    return shape_;
  }

  index_type entityDim() const noexcept
  {
    return entity_dim_;
  }

  index_type entityTag() const noexcept
  {
    return entity_tag_;
  }

  index_type physicalTag() const noexcept
  {
    return physical_tag_;
  }

  const std::string& physicalName() const noexcept
  {
    return physical_name_;
  }

private:
  std::vector<index_type> node_ids_;
  std::vector<Node>       nodes_;
  Shape                   shape_{Shape::Unknown};
  index_type              entity_dim_{0};
  index_type              entity_tag_{0};
  index_type              physical_tag_{0};
  std::string             physical_name_;
};

} // namespace femx
