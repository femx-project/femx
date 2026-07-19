#include <algorithm>
#include <stdexcept>

#include <femx/fem/Element.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{
namespace fem
{
HostGeometry makeGeometry(const Mesh& mesh)
{
  const Index dim       = mesh.dim();
  const Index num_nodes = mesh.numNodes();
  const Index num_elems = mesh.numElems();

  if (dim <= 0 || dim > 3)
  {
    throw std::runtime_error("Geometry mesh dimension must be between 1 and 3");
  }
  Index conn_size = 0;
  Index max_nodes = 0;
  for (Index ie = 0; ie < num_elems; ++ie)
  {
    const Element& elem = mesh.elem(ie);
    const Index    nn   = elem.nodeIds().size();
    if (nn <= 0)
    {
      throw std::runtime_error("Geometry elements must contain nodes");
    }
    conn_size += nn;
    max_nodes  = std::max(max_nodes, nn);

    for (Index in = 0; in < nn; ++in)
    {
      const Index node = elem.nodeIds()[in];
      if (node < 0 || node >= num_nodes)
      {
        throw std::runtime_error(
            "Geometry element connectivity is out of range");
      }
    }
  }

  HostGeometry geom;
  geom.dim_            = dim;
  geom.num_nodes_      = num_nodes;
  geom.num_elems_      = num_elems;
  geom.max_elem_nodes_ = max_nodes;
  geom.coords_.resize(num_nodes * dim);
  geom.elem_offsets_.resize(num_elems + 1);
  geom.conn_.resize(conn_size);

  for (Index node = 0; node < num_nodes; ++node)
  {
    for (Index d = 0; d < dim; ++d)
    {
      geom.coords_[node * dim + d] = mesh.node(node)[d];
    }
  }

  Index pos             = 0;
  geom.elem_offsets_[0] = 0;
  for (Index ie = 0; ie < num_elems; ++ie)
  {
    const auto& node_ids = mesh.elem(ie).nodeIds();
    for (Index in = 0; in < node_ids.size(); ++in)
    {
      geom.conn_[pos++] = node_ids[in];
    }
    geom.elem_offsets_[ie + 1] = pos;
  }

  return geom;
}

} // namespace fem
} // namespace femx
