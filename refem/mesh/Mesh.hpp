#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <refem/common/Types.hpp>
#include <refem/mesh/Cell.hpp>

namespace refem
{

class Mesh
{
public:
  using Node = Cell::Node;

  struct PhysicalName
  {
    index_type  dim = 0;
    index_type  tag = 0;
    std::string name;
  };

  struct BoundaryFacet
  {
    index_type              dim          = 0;
    index_type              entity_tag   = 0;
    index_type              physical_tag = 0;
    std::string             physical_name;
    Cell::Shape             shape = Cell::Shape::Unknown;
    std::vector<index_type> node_ids;
  };

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

  index_type numElems() const noexcept
  {
    return static_cast<index_type>(cells_.size());
  }

  const std::vector<Cell>& cells() const noexcept
  {
    return cells_;
  }

  const Cell& cell(index_type i) const
  {
    return cells_[static_cast<std::size_t>(i)];
  }

  const std::vector<BoundaryFacet>& boundaryFacets() const noexcept
  {
    return boundary_facets_;
  }

  std::vector<BoundaryFacet> boundaryFacets(const std::string& physical_name) const
  {
    std::vector<BoundaryFacet> facets;
    for (const auto& facet : boundary_facets_)
    {
      if (facet.physical_name == physical_name)
      {
        facets.push_back(facet);
      }
    }
    return facets;
  }

  const std::map<std::pair<index_type, index_type>, std::string>&
  physicalNames() const noexcept
  {
    return physical_names_;
  }

  std::string physicalName(index_type dim, index_type tag) const
  {
    const auto it = physical_names_.find({dim, tag});
    if (it == physical_names_.end())
    {
      return {};
    }
    return it->second;
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
    addCell(node_ids, Cell::Shape::Unknown, dim_, 0, 0, {});
  }

  void addCell(const std::vector<index_type>& node_ids,
               Cell::Shape                    shape,
               index_type                     entity_dim,
               index_type                     entity_tag,
               index_type                     physical_tag,
               std::string                    physical_name)
  {
    std::vector<Node> cell_nodes;
    cell_nodes.reserve(node_ids.size());
    for (index_type id : node_ids)
    {
      cell_nodes.push_back(node(id));
    }
    cells_.emplace_back(node_ids,
                        std::move(cell_nodes),
                        shape,
                        entity_dim,
                        entity_tag,
                        physical_tag,
                        std::move(physical_name));
  }

  void addBoundaryFacet(BoundaryFacet facet)
  {
    boundary_facets_.push_back(std::move(facet));
  }

  void addPhysicalName(index_type dim,
                       index_type tag,
                       std::string name)
  {
    physical_names_[{dim, tag}] = std::move(name);
  }

private:
  index_type                                             dim_{0};
  std::vector<Node>                                      nodes_;
  std::vector<Cell>                                      cells_;
  std::vector<BoundaryFacet>                             boundary_facets_;
  std::map<std::pair<index_type, index_type>, std::string>
      physical_names_;
};

} // namespace refem
