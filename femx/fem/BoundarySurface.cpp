#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#include <femx/fem/BoundarySurface.hpp>

namespace femx
{
namespace fem
{
namespace
{

constexpr Real min_measure = 1.0e-28;

HostVector<Mesh::BoundaryFacet> facetsByTag(const Mesh& mesh, Index physical_tag)
{
  HostVector<Mesh::BoundaryFacet> facets;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.ptag == physical_tag)
    {
      facets.push_back(facet);
    }
  }
  return facets;
}

void addEntry(SparseTripletMatrix& mat,
              Index                row,
              Index                col,
              Real                 val)
{
  mat.row_indices.push_back(row);
  mat.col_indices.push_back(col);
  mat.vals.push_back(val);
}

std::pair<Index, Index> orderedEdge(Index a, Index b)
{
  return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

void assembleSegment(const HostVector<Index>&  element,
                     const HostVector<Point3>& nodes,
                     BoundaryScalarMatrices&   out)
{
  if (element.size() != 2)
  {
    throw std::runtime_error("Boundary segment must have two nodes");
  }

  const Index i0     = element[0];
  const Index i1     = element[1];
  const Real  length = distance(nodes[i0], nodes[i1]);
  if (length <= min_measure)
  {
    throw std::runtime_error("Boundary surface contains a degenerate segment");
  }

  const Index ids[2]      = {i0, i1};
  const Real  mass_scale  = length / 6.0;
  const Real  stiff_scale = 1.0 / length;
  for (Index in = 0; in < 2; ++in)
  {
    out.load[ids[in]] += 0.5 * length;
    for (Index jn = 0; jn < 2; ++jn)
    {
      addEntry(out.mass,
               ids[in],
               ids[jn],
               mass_scale * (in == jn ? 2.0 : 1.0));
      addEntry(out.stiffness,
               ids[in],
               ids[jn],
               stiff_scale * (in == jn ? 1.0 : -1.0));
    }
  }
}

void assembleTriangle(const HostVector<Index>&  element,
                      const HostVector<Point3>& nodes,
                      BoundaryScalarMatrices&   out)
{
  if (element.size() != 3)
  {
    throw std::runtime_error("Boundary triangle must have three nodes");
  }

  const Point3& a    = nodes[element[0]];
  const Point3& b    = nodes[element[1]];
  const Point3& c    = nodes[element[2]];
  const Point3  e1   = difference(b, a);
  const Point3  e2   = difference(c, a);
  const Real    g00  = dot(e1, e1);
  const Real    g01  = dot(e1, e2);
  const Real    g11  = dot(e2, e2);
  const Real    det  = g00 * g11 - g01 * g01;
  const Real    area = triangleArea(a, b, c);
  if (det <= min_measure || area <= min_measure)
  {
    throw std::runtime_error("Boundary surface contains a degenerate triangle");
  }

  const Real inv_g[2][2] = {
      {g11 / det, -g01 / det},
      {-g01 / det, g00 / det}};
  const Real ref_grad[3][2] = {
      {-1.0, -1.0},
      {1.0, 0.0},
      {0.0, 1.0}};
  const Real mass_scale = area / 12.0;

  for (Index in = 0; in < 3; ++in)
  {
    const Index row  = element[in];
    out.load[row]   += area / 3.0;
    for (Index jn = 0; jn < 3; ++jn)
    {
      const Index col      = element[jn];
      Real        grad_dot = 0.0;
      for (Index id = 0; id < 2; ++id)
      {
        for (Index jd = 0; jd < 2; ++jd)
        {
          grad_dot +=
              ref_grad[in][id] * inv_g[id][jd] * ref_grad[jn][jd];
        }
      }
      addEntry(out.mass,
               row,
               col,
               mass_scale * (in == jn ? 2.0 : 1.0));
      addEntry(out.stiffness, row, col, area * grad_dot);
    }
  }
}

} // namespace

BoundarySurface::BoundarySurface(const Mesh&        mesh,
                                 const std::string& physical_name)
{
  initialize(mesh,
             mesh.boundaryFacets(physical_name),
             "physical boundary '" + physical_name + "'");
}

BoundarySurface::BoundarySurface(const Mesh& mesh, Index physical_tag)
{
  initialize(mesh,
             facetsByTag(mesh, physical_tag),
             "physical boundary tag " + std::to_string(physical_tag));
}

void BoundarySurface::initialize(
    const Mesh&                            mesh,
    const HostVector<Mesh::BoundaryFacet>& facets,
    const std::string&                     label)
{
  if (facets.empty())
  {
    throw std::runtime_error("No facets found for " + label);
  }

  dim_ = facets.front().dim;
  std::map<Index, Index> local_ids;
  for (const auto& facet : facets)
  {
    if (facet.dim != dim_)
    {
      throw std::runtime_error(label + " contains mixed facet dimensions");
    }
    if (facet.shape != ElementShape::Segment
        && facet.shape != ElementShape::Triangle)
    {
      throw std::runtime_error(
          label + " supports only segment and triangle facets");
    }

    HostVector<Index> element;
    element.reserve(facet.nids.size());
    for (Index mesh_node_id : facet.nids)
    {
      auto it = local_ids.find(mesh_node_id);
      if (it == local_ids.end())
      {
        const Index local_id = nodes_.size();
        it                   = local_ids.emplace(mesh_node_id, local_id).first;
        mesh_node_ids_.push_back(mesh_node_id);
        nodes_.push_back(mesh.node(mesh_node_id));
      }
      element.push_back(it->second);
    }
    elements_.push_back(std::move(element));
    element_shapes_.push_back(facet.shape);
  }

  findRimNodes();
}

void BoundarySurface::findRimNodes()
{
  HostVector<Index> is_rim(numNodes(), 0);
  if (dim_ == 1)
  {
    HostVector<Index> incidence(numNodes(), 0);
    for (Index ie = 0; ie < numElements(); ++ie)
    {
      if (element_shapes_[ie] != ElementShape::Segment
          || elements_[ie].size() != 2)
      {
        throw std::runtime_error("One-dimensional boundary surface must use segments");
      }
      ++incidence[elements_[ie][0]];
      ++incidence[elements_[ie][1]];
    }
    for (Index in = 0; in < numNodes(); ++in)
    {
      is_rim[in] = incidence[in] != 2;
    }
  }
  else if (dim_ == 2)
  {
    std::map<std::pair<Index, Index>, Index> edge_counts;
    for (Index ie = 0; ie < numElements(); ++ie)
    {
      if (element_shapes_[ie] != ElementShape::Triangle
          || elements_[ie].size() != 3)
      {
        throw std::runtime_error("Two-dimensional boundary surface must use triangles");
      }
      const auto& elem = elements_[ie];
      ++edge_counts[orderedEdge(elem[0], elem[1])];
      ++edge_counts[orderedEdge(elem[1], elem[2])];
      ++edge_counts[orderedEdge(elem[2], elem[0])];
    }
    for (const auto& edge : edge_counts)
    {
      if (edge.second == 1)
      {
        is_rim[edge.first.first]  = true;
        is_rim[edge.first.second] = true;
      }
      else if (edge.second != 2)
      {
        throw std::runtime_error("Boundary surface is non-manifold");
      }
    }
  }
  else
  {
    throw std::runtime_error("Boundary surface dimension must be one or two");
  }

  for (Index in = 0; in < numNodes(); ++in)
  {
    if (is_rim[in])
    {
      rim_node_ids_.push_back(in);
    }
  }
}

BoundaryScalarMatrices BoundarySurface::scalarMatrices() const
{
  BoundaryScalarMatrices out;
  out.stiffness.rows = numNodes();
  out.stiffness.cols = numNodes();
  out.mass.rows      = numNodes();
  out.mass.cols      = numNodes();
  out.load.resize(numNodes());

  for (Index ie = 0; ie < numElements(); ++ie)
  {
    if (element_shapes_[ie] == ElementShape::Segment)
    {
      assembleSegment(elements_[ie], nodes_, out);
    }
    else if (element_shapes_[ie] == ElementShape::Triangle)
    {
      assembleTriangle(elements_[ie], nodes_, out);
    }
    else
    {
      throw std::runtime_error("Unsupported boundary surface element shape");
    }
  }
  return out;
}

} // namespace fem
} // namespace femx
