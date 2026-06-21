#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/fem/Cell.hpp>

namespace femx
{

class Mesh
{
public:
  using Node = Cell::Node;

  struct PhysicalName
  {
    Index       dim = 0;
    Index       tag = 0;
    std::string name;
  };

  struct BoundaryFacet
  {
    Index              dim          = 0;
    Index              entity_tag   = 0;
    Index              physical_tag = 0;
    std::string        physical_name;
    Cell::Shape        shape = Cell::Shape::Unknown;
    std::vector<Index> node_ids;
  };

  Mesh() = default;

  explicit Mesh(Index dim)
    : dim_(dim)
  {
  }

  static Mesh makeStructuredQuad(Index nx,
                                 Index ny,
                                 Real  x_min = 0.0,
                                 Real  x_max = 1.0,
                                 Real  y_min = 0.0,
                                 Real  y_max = 1.0);

  Index dim() const noexcept
  {
    return dim_;
  }

  Index numNodes() const noexcept
  {
    return static_cast<Index>(nodes_.size());
  }

  Index numElems() const noexcept
  {
    return static_cast<Index>(cells_.size());
  }

  const std::vector<Cell>& cells() const noexcept
  {
    return cells_;
  }

  const Cell& cell(Index ic) const
  {
    return cells_[static_cast<std::size_t>(ic)];
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

  const std::map<std::pair<Index, Index>, std::string>&
  physicalNames() const noexcept
  {
    return physical_names_;
  }

  std::string physicalName(Index dim, Index tag) const
  {
    const auto it = physical_names_.find({dim, tag});
    if (it == physical_names_.end())
    {
      return {};
    }
    return it->second;
  }

  const Node& node(Index in) const
  {
    return nodes_[static_cast<std::size_t>(in)];
  }

  const Index* cellNodeIds(Index ic) const
  {
    return cells_[static_cast<std::size_t>(ic)].nodeIdsData();
  }

  void addNode(const Node& node)
  {
    nodes_.push_back(node);
  }

  void addCell(const std::vector<Index>& node_ids)
  {
    addCell(node_ids, Cell::Shape::Unknown, dim_, 0, 0, {});
  }

  void addCell(const std::vector<Index>& node_ids,
               Cell::Shape               shape,
               Index                     entity_dim,
               Index                     entity_tag,
               Index                     physical_tag,
               std::string               physical_name)
  {
    std::vector<Node> cell_nodes;
    cell_nodes.reserve(node_ids.size());
    for (Index in : node_ids)
    {
      cell_nodes.push_back(node(in));
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

  void addPhysicalName(Index       dim,
                       Index       tag,
                       std::string name)
  {
    physical_names_[{dim, tag}] = std::move(name);
  }

private:
  Index                      dim_{0};
  std::vector<Node>          nodes_;
  std::vector<Cell>          cells_;
  std::vector<BoundaryFacet> boundary_facets_;
  std::map<std::pair<Index, Index>, std::string>
      physical_names_;
};

} // namespace femx
