#pragma once

#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/Element.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

/** Sparse matrix entries in coordinate (COO) form. */
struct SparseTripletMatrix
{
  Index             rows{0};
  Index             cols{0};
  HostVector<Index> row_indices;
  HostVector<Index> col_indices;
  HostVector<Real>  vals;
};

/** Scalar P1 mass, stiffness, and constant-load data on a boundary surface. */
struct BoundaryScalarMatrices
{
  SparseTripletMatrix stiffness;
  SparseTripletMatrix mass;
  HostVector<Real>    load;
};

/**
 * @brief A simplicial boundary submesh with local node numbering.
 *
 * BoundarySurface extracts segment or triangle facets from a Mesh physical
 * boundary.  It preserves the corresponding volume-mesh node ids while
 * providing compact local connectivity for surface computations.
 */
class BoundarySurface
{
public:
  BoundarySurface(const Mesh& mesh, const std::string& physical_name);
  BoundarySurface(const Mesh& mesh, Index physical_tag);

  Index dim() const noexcept
  {
    return dim_;
  }

  Index numNodes() const noexcept
  {
    return nodes_.size();
  }

  Index numElements() const noexcept
  {
    return elements_.size();
  }

  const HostVector<Index>& meshNodeIds() const noexcept
  {
    return mesh_node_ids_;
  }

  const HostVector<Point3>& nodes() const noexcept
  {
    return nodes_;
  }

  const HostVector<HostVector<Index>>& elements() const noexcept
  {
    return elements_;
  }

  const HostVector<Element::Shape>& elementShapes() const noexcept
  {
    return element_shapes_;
  }

  /** Local node ids on the boundary of this boundary surface. */
  const HostVector<Index>& rimNodeIds() const noexcept
  {
    return rim_node_ids_;
  }

  /** Assemble scalar linear finite-element matrices on the surface. */
  BoundaryScalarMatrices scalarMatrices() const;

private:
  void initialize(const Mesh&                            mesh,
                  const HostVector<Mesh::BoundaryFacet>& facets,
                  const std::string&                     label);
  void findRimNodes();

private:
  Index                         dim_{0};
  HostVector<Index>             mesh_node_ids_;
  HostVector<Point3>            nodes_;
  HostVector<HostVector<Index>> elements_;
  HostVector<Element::Shape>    element_shapes_;
  HostVector<Index>             rim_node_ids_;
};

} // namespace fem
} // namespace femx
