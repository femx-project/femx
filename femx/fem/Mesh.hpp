#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/fem/Element.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class Mesh
{
public:
  using Node = Element::Node;

  struct BoundaryFacet
  {
    Index          dim  = 0;
    Index          etag = 0;
    Index          ptag = 0;
    std::string    pname;
    Element::Shape shape = Element::Shape::Unknown;
    Vector<Index>  nids;
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
    return nodes_.size();
  }

  Index numElems() const noexcept
  {
    return elems_.size();
  }

  const Vector<Element>& elems() const noexcept
  {
    return elems_;
  }

  const Element& elem(Index ie) const
  {
    return elems_[ie];
  }

  const Vector<BoundaryFacet>& boundaryFacets() const noexcept
  {
    return boundary_facets_;
  }

  Vector<BoundaryFacet> boundaryFacets(const std::string& pname) const
  {
    Vector<BoundaryFacet> facets;
    for (const auto& facet : boundary_facets_)
    {
      if (facet.pname == pname)
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
    return nodes_[in];
  }

  const Index* elemNodeIds(Index ie) const
  {
    return elems_[ie].nodeIdsData();
  }

  void addNode(const Node& node)
  {
    nodes_.push_back(node);
  }

  void addElem(const Vector<Index>& nids)
  {
    addElem(nids, Element::Shape::Unknown, dim_, 0, 0, {});
  }

  void addElem(const Vector<Index>& nids,
               Element::Shape       shape,
               Index                edim,
               Index                etag,
               Index                ptag,
               std::string          pname)
  {
    Vector<Node> cn;
    cn.reserve(nids.size());
    for (Index in : nids)
    {
      cn.push_back(node(in));
    }
    elems_.emplace_back(nids,
                        std::move(cn),
                        shape,
                        edim,
                        etag,
                        ptag,
                        std::move(pname));
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
  Index                 dim_{0};
  Vector<Node>          nodes_;
  Vector<Element>       elems_;
  Vector<BoundaryFacet> boundary_facets_;
  std::map<std::pair<Index, Index>, std::string>
      physical_names_;
};

} // namespace femx
