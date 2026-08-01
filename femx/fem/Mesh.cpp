#include <algorithm>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Context.hpp>

namespace femx
{
namespace fem
{

MeshView<MemorySpace::Host> Mesh::view() const
{
  require(dim_ > 0 && dim_ <= 3,
          "Mesh dimension must be between one and three");
  require(coords_.size() == numNodes() * dim_,
          "Mesh coordinate storage is inconsistent");
  require(elem_offsets_.size() == numElems() + 1
              && elem_offsets_.back() == conn_.size(),
          "Mesh connectivity storage is inconsistent");
  require(elem_entity_dims_.size() == numElems()
              && elem_entity_tags_.size() == numElems()
              && elem_physical_tags_.size() == numElems(),
          "Mesh element classification storage is inconsistent");
  return {dim_,
          numNodes(),
          numElems(),
          max_elem_nodes_,
          coords_.view(),
          elem_offsets_.view(),
          conn_.view()};
}

void Mesh::addNode(const Node& node)
{
  nodes_.push_back(node);
  if (dim_ > 0 && dim_ <= 3)
  {
    for (Index id = 0; id < dim_; ++id)
    {
      coords_.push_back(node[id]);
    }
  }
}

void Mesh::addElem(const HostVector<Index>& nids,
                   ElementShape             shape,
                   Index                    edim,
                   Index                    etag,
                   Index                    ptag,
                   std::string              pname)
{
  require(!nids.empty(),
          "Mesh elements must contain nodes");

  for (Index in : nids)
  {
    require(in >= 0 && in < numNodes(),
            "Mesh element connectivity is out of range");
  }

  for (Index in : nids)
  {
    conn_.push_back(in);
  }
  elem_offsets_.push_back(conn_.size());
  elem_shapes_.push_back(shape);
  elem_entity_dims_.push_back(edim);
  elem_entity_tags_.push_back(etag);
  elem_physical_tags_.push_back(ptag);
  if (!pname.empty())
  {
    addPhysicalName(edim, ptag, std::move(pname));
  }
  max_elem_nodes_ = std::max(max_elem_nodes_, nids.size());
}

void copy(const Mesh&                           src,
          DeviceMesh&                           dst,
          linalg::Context<MemorySpace::Device>& ctx)
{
  (void) src.view();
  dst.dim_            = src.dim_;
  dst.num_nodes_      = src.numNodes();
  dst.num_elems_      = src.numElems();
  dst.max_elem_nodes_ = src.max_elem_nodes_;

  auto& vec_handler = ctx.vectorHandler();
  vec_handler.copy(src.coords_, dst.coords_);
  vec_handler.copy(src.elem_offsets_, dst.elem_offsets_);
  vec_handler.copy(src.conn_, dst.conn_);
}

Mesh Mesh::makeStructuredQuad(Index num_x_cells,
                              Index num_y_cells,
                              Real  x_min,
                              Real  x_max,
                              Real  y_min,
                              Real  y_max)
{
  Mesh mesh(2);

  const Real dx = (x_max - x_min) / static_cast<Real>(num_x_cells);
  const Real dy = (y_max - y_min) / static_cast<Real>(num_y_cells);

  for (Index j = 0; j <= num_y_cells; ++j)
  {
    const Real y = y_min + static_cast<Real>(j) * dy;
    for (Index i = 0; i <= num_x_cells; ++i)
    {
      const Real x = x_min + static_cast<Real>(i) * dx;
      mesh.addNode({x, y, 0.0});
    }
  }

  const Index nodes_per_row = num_x_cells + 1;
  for (Index j = 0; j < num_y_cells; ++j)
  {
    for (Index i = 0; i < num_x_cells; ++i)
    {
      const Index n0 = j * nodes_per_row + i;
      const Index n1 = n0 + 1;
      const Index n3 = n0 + nodes_per_row;
      const Index n2 = n3 + 1;
      mesh.addElem({n0, n1, n2, n3},
                   ElementShape::Quadrilateral,
                   2,
                   0,
                   0,
                   {});
    }
  }

  return mesh;
}

} // namespace fem
} // namespace femx
