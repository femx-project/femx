#pragma once

#include <array>
#include <utility>
#include <vector>

#include <refem/common/Types.hpp>

namespace refem
{

class Cell
{
public:
  using Node = std::array<real_type, 3>;

  Cell() = default;

  Cell(std::vector<index_type> node_ids,
       std::vector<Node>       nodes)
    : node_ids_(std::move(node_ids)),
      nodes_(std::move(nodes))
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

private:
  std::vector<index_type> node_ids_;
  std::vector<Node>       nodes_;
};

} // namespace refem
