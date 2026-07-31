#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx
{
namespace linalg
{
class CudaContext;
}

namespace fem
{

class Mesh;
class DeviceMesh;

/** @brief Identify the topology of a mesh element. */
enum class ElementShape
{
  Unknown,
  Segment,
  Triangle,
  Quadrilateral,
  Tetrahedron,
  Hexahedron
};

/** @brief Provide lightweight access to mesh coordinates and connectivity. */
template <MemorySpace Space>
class MeshView
{
public:
  FEMX_HOST_DEVICE MeshView() = default;

  /**
   * @brief Construct a view over flattened mesh data.
   *
   * @param[in] dim - Spatial dimension.
   * @param[in] num_nodes - Number of global nodes.
   * @param[in] num_elems - Number of elements.
   * @param[in] max_elem_nodes - Largest element node count.
   * @param[in] coords - Node-major coordinate components.
   * @param[in] elem_offsets - Element offsets into `conn`.
   * @param[in] conn - Flattened element-to-node connectivity.
   */
  FEMX_HOST_DEVICE MeshView(
      Index                          dim,
      Index                          num_nodes,
      Index                          num_elems,
      Index                          max_elem_nodes,
      VectorView<Space, const Real>  coords,
      VectorView<Space, const Index> elem_offsets,
      VectorView<Space, const Index> conn)
    : dim_(dim),
      num_nodes_(num_nodes),
      num_elems_(num_elems),
      max_elem_nodes_(max_elem_nodes),
      coords_(coords),
      elem_offsets_(elem_offsets),
      conn_(conn)
  {
  }

  /** @brief Return the spatial dimension. */
  FEMX_HOST_DEVICE Index dim() const
  {
    return dim_;
  }

  /** @brief Return the number of global nodes. */
  FEMX_HOST_DEVICE Index numNodes() const
  {
    return num_nodes_;
  }

  /** @brief Return the number of elements. */
  FEMX_HOST_DEVICE Index numElems() const
  {
    return num_elems_;
  }

  /** @brief Return the largest element node count. */
  FEMX_HOST_DEVICE Index maxElemNodes() const
  {
    return max_elem_nodes_;
  }

  /**
   * @brief Return one coordinate component for a global node.
   *
   * @param[in] node - Global node index.
   * @param[in] d - Coordinate component.
   * @return Requested coordinate component.
   */
  FEMX_HOST_DEVICE Real coord(Index node, Index d) const
  {
    return coords_[node * dim_ + d];
  }

  /**
   * @brief Return the number of nodes on one element.
   *
   * @param[in] ie - Element index.
   * @return Number of nodes on the element.
   */
  FEMX_HOST_DEVICE Index elemNumNodes(Index ie) const
  {
    return elem_offsets_[ie + 1] - elem_offsets_[ie];
  }

  /**
   * @brief Map one element-local node to a global node.
   *
   * @param[in] ie - Element index.
   * @param[in] in - Element-local node index.
   * @return Global node index.
   */
  FEMX_HOST_DEVICE Index elemNodeId(Index ie, Index in) const
  {
    return conn_[elem_offsets_[ie] + in];
  }

private:
  Index                          dim_{0};            ///< Spatial dimension.
  Index                          num_nodes_{0};      ///< Number of global nodes.
  Index                          num_elems_{0};      ///< Number of elements.
  Index                          max_elem_nodes_{0}; ///< Largest element node count.
  VectorView<Space, const Real>  coords_;            ///< Node-major coordinates.
  VectorView<Space, const Index> elem_offsets_;      ///< Element connectivity offsets.
  VectorView<Space, const Index> conn_;              ///< Flattened connectivity.
};

/**
 * @brief Own mesh coordinates and connectivity in Device memory.
 *
 * Construct this execution representation by copying a Host `Mesh`.
 */
class DeviceMesh
{
public:
  DeviceMesh() = default;

  DeviceMesh(const DeviceMesh&)                = delete;
  DeviceMesh(DeviceMesh&&) noexcept            = default;
  DeviceMesh& operator=(const DeviceMesh&)     = delete;
  DeviceMesh& operator=(DeviceMesh&&) noexcept = default;

  /** @brief Return the spatial dimension. */
  Index dim() const noexcept
  {
    return dim_;
  }

  /** @brief Return the number of global nodes. */
  Index numNodes() const noexcept
  {
    return num_nodes_;
  }

  /** @brief Return the number of elements. */
  Index numElems() const noexcept
  {
    return num_elems_;
  }

  /** @brief Return the largest element node count. */
  Index maxElemNodes() const noexcept
  {
    return max_elem_nodes_;
  }

  /** @brief Return a non-owning view valid while this object is alive. */
  MeshView<MemorySpace::Device> view() const noexcept
  {
    return {dim_,
            num_nodes_,
            num_elems_,
            max_elem_nodes_,
            coords_.view(),
            elem_offsets_.view(),
            conn_.view()};
  }

private:
  friend void copy(const Mesh&          src,
                   DeviceMesh&          dst,
                   linalg::CudaContext& ctx);

  Index               dim_{0};            ///< Spatial dimension.
  Index               num_nodes_{0};      ///< Number of global nodes.
  Index               num_elems_{0};      ///< Number of elements.
  Index               max_elem_nodes_{0}; ///< Largest element node count.
  DeviceVector<Real>  coords_;            ///< Node-major coordinates.
  DeviceVector<Index> elem_offsets_;      ///< Element connectivity offsets.
  DeviceVector<Index> conn_;              ///< Flattened connectivity.
};

/**
 * @brief Unstructured finite-element mesh with optional physical boundary data.
 *
 * Mesh stores nodal coordinates, element connectivity, and the physical names
 * imported from mesh files or generated by structured-mesh helpers. It also
 * provides flat Host and Device views for assembly. Element connectivity and
 * classification are stored directly by the mesh without per-element copies.
 */
class Mesh
{
public:
  /** @brief Three-component storage for a mesh node coordinate. */
  using Node = std::array<Real, 3>;

  /**
   * @brief Boundary facet imported from a mesh file.
   *
   * Facets are grouped by physical tag/name and can be used to impose boundary
   * conditions on named parts of the mesh.
   */
  struct BoundaryFacet
  {
    Index             dim  = 0;                      ///< Topological dimension of the facet.
    Index             etag = 0;                      ///< Element tag from the mesh file.
    Index             ptag = 0;                      ///< Physical boundary tag.
    std::string       pname;                         ///< Physical boundary name.
    ElementShape      shape = ElementShape::Unknown; ///< Facet element shape.
    HostVector<Index> nids;                          ///< Mesh-node ids on the facet.
  };

  Mesh() = default;

  /**
   * @brief Construct an empty mesh with a spatial dimension.
   *
   * @param[in] dim - Spatial dimension.
   */
  explicit Mesh(Index dim)
    : dim_(dim)
  {
  }

  /**
   * @brief Create a two-dimensional quadrilateral mesh on a rectangle.
   *
   * @param[in] num_x_cells - Number of cells in the x direction.
   * @param[in] num_y_cells - Number of cells in the y direction.
   * @param[in] x_min - Lower x coordinate.
   * @param[in] x_max - Upper x coordinate.
   * @param[in] y_min - Lower y coordinate.
   * @param[in] y_max - Upper y coordinate.
   * @return Mesh with Q1-compatible quadrilateral connectivity.
   */
  static Mesh makeStructuredQuad(Index num_x_cells,
                                 Index num_y_cells,
                                 Real  x_min = 0.0,
                                 Real  x_max = 1.0,
                                 Real  y_min = 0.0,
                                 Real  y_max = 1.0);

  /** @brief Return the spatial dimension. */
  Index dim() const noexcept
  {
    return dim_;
  }

  /** @brief Return the number of mesh nodes. */
  Index numNodes() const noexcept
  {
    return nodes_.size();
  }

  /** @brief Return the number of mesh elements. */
  Index numElems() const noexcept
  {
    return elem_shapes_.size();
  }

  /** @brief Return the largest element node count. */
  Index maxElemNodes() const noexcept
  {
    return max_elem_nodes_;
  }

  /**
   * @brief Return a non-owning Host view of coordinates and connectivity.
   *
   * @return View valid until nodes or elements are added.
   * @throws std::runtime_error - If the mesh dimension or internal storage is
   * invalid.
   */
  MeshView<MemorySpace::Host> view() const;

  /**
   * @brief Return the number of nodes on one element.
   *
   * @param[in] ie - Element index.
   * @return Number of nodes on the element.
   */
  Index elemNumNodes(Index ie) const noexcept
  {
    return elem_offsets_[ie + 1] - elem_offsets_[ie];
  }

  /**
   * @brief Return the node identifiers of one element.
   *
   * @param[in] ie - Element index.
   * @return View of the element node identifiers.
   */
  HostVectorView<const Index> elemNodeIds(Index ie) const
  {
    const Index offset = elem_offsets_[ie];
    return {conn_.data() + offset, elemNumNodes(ie)};
  }

  /**
   * @brief Map one element-local node to a mesh node identifier.
   *
   * @param[in] ie - Element index.
   * @param[in] in - Element-local node index.
   * @return Mesh node identifier.
   */
  Index elemNodeId(Index ie, Index in) const noexcept
  {
    return conn_[elem_offsets_[ie] + in];
  }

  /**
   * @brief Return one element-local node coordinate.
   *
   * @param[in] ie - Element index.
   * @param[in] in - Element-local node index.
   * @return Node coordinate.
   */
  const Node& elemNode(Index ie, Index in) const
  {
    return node(elemNodeId(ie, in));
  }

  /**
   * @brief Return the topology of one element.
   *
   * @param[in] ie - Element index.
   * @return Element topology.
   */
  ElementShape elemShape(Index ie) const noexcept
  {
    return elem_shapes_[ie];
  }

  /**
   * @brief Return the classified entity dimension of one element.
   *
   * @param[in] ie - Element index.
   * @return Entity dimension.
   */
  Index elemEntityDim(Index ie) const noexcept
  {
    return elem_entity_dims_[ie];
  }

  /**
   * @brief Return the classified entity tag of one element.
   *
   * @param[in] ie - Element index.
   * @return Entity tag.
   */
  Index elemEntityTag(Index ie) const noexcept
  {
    return elem_entity_tags_[ie];
  }

  /**
   * @brief Return the physical tag of one element.
   *
   * @param[in] ie - Element index.
   * @return Physical tag.
   */
  Index elemPhysicalTag(Index ie) const noexcept
  {
    return elem_physical_tags_[ie];
  }

  /**
   * @brief Return the physical name of one element.
   *
   * @param[in] ie - Element index.
   * @return Physical name, or an empty string if none exists.
   */
  std::string elemPhysicalName(Index ie) const
  {
    return physicalName(elemEntityDim(ie), elemPhysicalTag(ie));
  }

  /** @brief Return all classified boundary facets. */
  const HostVector<BoundaryFacet>& boundaryFacets() const noexcept
  {
    return boundary_facets_;
  }

  /**
   * @brief Return boundary facets with a physical name.
   *
   * @param[in] pname - Physical boundary name.
   * @return Matching boundary facets.
   */
  HostVector<BoundaryFacet> boundaryFacets(const std::string& pname) const
  {
    HostVector<BoundaryFacet> facets;
    for (const auto& facet : boundary_facets_)
    {
      if (facet.pname == pname)
      {
        facets.push_back(facet);
      }
    }
    return facets;
  }

  /** @brief Return physical names keyed by dimension and tag. */
  const std::map<std::pair<Index, Index>, std::string>&
  physicalNames() const noexcept
  {
    return physical_names_;
  }

  /**
   * @brief Return the physical name for a dimension and tag.
   *
   * @param[in] dim - Entity dimension.
   * @param[in] tag - Physical tag.
   * @return Physical name, or an empty string if none exists.
   */
  std::string physicalName(Index dim, Index tag) const
  {
    const auto it = physical_names_.find({dim, tag});
    if (it == physical_names_.end())
    {
      return {};
    }
    return it->second;
  }

  /**
   * @brief Return one mesh node.
   *
   * @param[in] in - Node index.
   * @return Node at `in`.
   */
  const Node& node(Index in) const
  {
    return nodes_[in];
  }

  /**
   * @brief Add one mesh node.
   *
   * @param[in] node - Node coordinates.
   */
  void addNode(const Node& node);

  /**
   * @brief Add one element using mesh node identifiers.
   *
   * @param[in] nids - Mesh node identifiers in element-local order.
   */
  void addElem(const HostVector<Index>& nids)
  {
    addElem(nids, ElementShape::Unknown, dim_, 0, 0, {});
  }

  /**
   * @brief Add one element with topology and physical classification.
   *
   * @param[in] nids - Mesh node identifiers in element-local order.
   * @param[in] shape - Element topology.
   * @param[in] edim - Entity dimension from the mesh generator.
   * @param[in] etag - Entity tag from the mesh generator.
   * @param[in] ptag - Physical group tag.
   * @param[in] pname - Physical group name.
   */
  void addElem(const HostVector<Index>& nids,
               ElementShape             shape,
               Index                    edim,
               Index                    etag,
               Index                    ptag,
               std::string              pname);

  /**
   * @brief Add a classified boundary facet.
   *
   * @param[in] facet - Boundary facet.
   */
  void addBoundaryFacet(BoundaryFacet facet)
  {
    boundary_facets_.push_back(std::move(facet));
  }

  /**
   * @brief Add a physical name.
   *
   * @param[in] dim - Entity dimension.
   * @param[in] tag - Physical tag.
   * @param[in] name - Physical name.
   */
  void addPhysicalName(Index       dim,
                       Index       tag,
                       std::string name)
  {
    physical_names_[{dim, tag}] = std::move(name);
  }

private:
  friend void copy(const Mesh&          src,
                   DeviceMesh&          dst,
                   linalg::CudaContext& ctx);

  Index                     dim_{0};             ///< Spatial dimension.
  HostVector<Node>          nodes_;              ///< Global nodes.
  Index                     max_elem_nodes_{0};  ///< Largest element node count.
  HostVector<Real>          coords_;             ///< Flat node-major coordinates.
  HostVector<Index>         elem_offsets_{0};    ///< Element connectivity offsets.
  HostVector<Index>         conn_;               ///< Flattened connectivity.
  HostVector<ElementShape>  elem_shapes_;        ///< Element topologies.
  HostVector<Index>         elem_entity_dims_;   ///< Element entity dimensions.
  HostVector<Index>         elem_entity_tags_;   ///< Element entity tags.
  HostVector<Index>         elem_physical_tags_; ///< Element physical tags.
  HostVector<BoundaryFacet> boundary_facets_;    ///< Classified boundary facets.
  std::map<std::pair<Index, Index>, std::string>
      physical_names_; ///< Physical names by dimension and tag.
};

/**
 * @brief Copy a Host mesh to Device-owned execution storage.
 *
 * @param[in] src - Host mesh kept alive while copies are queued.
 * @param[out] dst - Device mesh replaced by the copy.
 * @param[in] ctx - CUDA context used for the asynchronous copy.
 */
void copy(const Mesh&          src,
          DeviceMesh&          dst,
          linalg::CudaContext& ctx);

} // namespace fem
} // namespace femx
