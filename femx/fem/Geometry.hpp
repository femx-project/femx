#pragma once

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx
{
namespace fem
{

class Mesh;

template <MemorySpace Space>
class Geometry;

/** @brief Flatten mesh coordinates and connectivity into host backend storage. */
Geometry<MemorySpace::Host> makeGeometry(const Mesh& mesh);

/**
 * @brief Copy host geometry to device-owned storage.
 * @param src Host geometry kept alive while copies are queued.
 * @param dst Device geometry replaced by the copy.
 * @param ctx CUDA stream on which copies are enqueued.
 */
void copy(const Geometry<MemorySpace::Host>& src,
          Geometry<MemorySpace::Device>&     dst,
          CudaContext&                       ctx);

/** @brief Lightweight access to flattened mesh geometry. */
template <MemorySpace Space>
class GeometryView
{
public:
  FEMX_HOST_DEVICE GeometryView() = default;

  /** @brief Construct a view over flattened coordinates and connectivity. */
  FEMX_HOST_DEVICE GeometryView(
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

  /** @brief Return the number of global geometry nodes. */
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

  /** @brief Coordinate component for a valid global node and axis. */
  FEMX_HOST_DEVICE Real coord(Index node, Index d) const
  {
    return coords_[node * dim_ + d];
  }

  /** @brief Return the number of geometry nodes on element `ie`. */
  FEMX_HOST_DEVICE Index elemNumNodes(Index ie) const
  {
    return elem_offsets_[ie + 1] - elem_offsets_[ie];
  }

  /** @brief Map local node `in` of element `ie` to a global node. */
  FEMX_HOST_DEVICE Index elemNode(Index ie, Index in) const
  {
    return conn_[elem_offsets_[ie] + in];
  }

private:
  Index                          dim_{0};
  Index                          num_nodes_{0};
  Index                          num_elems_{0};
  Index                          max_elem_nodes_{0};
  VectorView<Space, const Real>  coords_;
  VectorView<Space, const Index> elem_offsets_;
  VectorView<Space, const Index> conn_;
};

/**
 * @brief Memory-space-owned coordinates and element connectivity.
 *
 * Coordinates are node-major with `dim()` entries per node. Connectivity is
 * element-major and uses `elemOffsets()` to support runtime local node counts.
 */
template <MemorySpace Space>
class Geometry
{
public:
  Geometry() = default;

  Geometry(const Geometry&)                = default;
  Geometry(Geometry&&) noexcept            = default;
  Geometry& operator=(const Geometry&)     = default;
  Geometry& operator=(Geometry&&) noexcept = default;

  /** @brief Return the spatial dimension. */
  Index dim() const noexcept
  {
    return dim_;
  }

  /** @brief Return the number of global geometry nodes. */
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

  /** @brief Return a non-owning kernel view valid while this object is alive. */
  GeometryView<Space> view() const noexcept
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
  friend Geometry<MemorySpace::Host> makeGeometry(const Mesh& mesh);

  friend void copy(const Geometry<MemorySpace::Host>& src,
                   Geometry<MemorySpace::Device>&     dst,
                   CudaContext&                       ctx);

  Index                dim_{0};
  Index                num_nodes_{0};
  Index                num_elems_{0};
  Index                max_elem_nodes_{0};
  Vector<Space, Real>  coords_;
  Vector<Space, Index> elem_offsets_;
  Vector<Space, Index> conn_;
};

using HostGeometry   = Geometry<MemorySpace::Host>;
using DeviceGeometry = Geometry<MemorySpace::Device>;

inline void copy(const Geometry<MemorySpace::Host>& src,
                 Geometry<MemorySpace::Device>&     dst,
                 CudaContext&                       ctx)
{
  linalg::CudaVectorHandler vec_handler(ctx);
  dst.dim_            = src.dim_;
  dst.num_nodes_      = src.num_nodes_;
  dst.num_elems_      = src.num_elems_;
  dst.max_elem_nodes_ = src.max_elem_nodes_;
  vec_handler.copy(src.coords_, dst.coords_);
  vec_handler.copy(src.elem_offsets_, dst.elem_offsets_);
  vec_handler.copy(src.conn_, dst.conn_);
}

} // namespace fem
} // namespace femx
