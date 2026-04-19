#pragma once

#include <array>
#include <vector>

#include <refem/common/Types.hpp>
#include <refem/mesh/Cell.hpp>

namespace refem
{

class Mesh
{
public:
  using Node = Cell::Node;

  Mesh() = default;

  explicit Mesh(index_type dim)
    : dim_(dim)
  {
  }

  static Mesh makeStructuredQuad(index_type nx,
                                 index_type ny,
                                 real_type  x_min = 0.0,
                                 real_type  x_max = 1.0,
                                 real_type  y_min = 0.0,
                                 real_type  y_max = 1.0);

  index_type dim() const noexcept
  {
    return dim_;
  }

  index_type numNodes() const noexcept
  {
    return static_cast<index_type>(nodes_.size());
  }

  index_type numCells() const noexcept
  {
    return static_cast<index_type>(cells_.size());
  }

  const std::vector<Cell>& cells() const noexcept
  {
    return cells_;
  }

  const Node& node(index_type i) const
  {
    return nodes_[static_cast<std::size_t>(i)];
  }

  const index_type* cellNodeIds(index_type cell) const
  {
    return cells_[static_cast<std::size_t>(cell)].nodeIdsData();
  }

  void addNode(const Node& node)
  {
    nodes_.push_back(node);
  }

  void addCell(const std::vector<index_type>& node_ids)
  {
    std::vector<Node> cell_nodes;
    cell_nodes.reserve(node_ids.size());
    for (index_type id : node_ids)
    {
      cell_nodes.push_back(node(id));
    }
    cells_.emplace_back(node_ids, std::move(cell_nodes));
  }

private:
  index_type        dim_{0};
  std::vector<Node> nodes_;
  std::vector<Cell> cells_;
};

} // namespace refem
