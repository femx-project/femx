#include "StateFields.hpp"

#include <stdexcept>

using namespace femx;

namespace femx::model::ns
{

void splitStateFields(HostVectorView<const Real> state,
                      const fem::MixedFESpace&   space,
                      HostVector<Real>&          ux,
                      HostVector<Real>&          uy,
                      HostVector<Real>&          uz,
                      HostVector<Real>&          pressure)
{
  const fem::Mesh& mesh      = space.mesh();
  const Index      num_nodes = mesh.numNodes();
  if (state.size() != space.numDofs())
  {
    throw std::runtime_error(
        "Navier-Stokes state size does not match the finite-element space");
  }
  if (ux.size() != num_nodes || uy.size() != num_nodes
      || uz.size() != num_nodes || pressure.size() != num_nodes)
  {
    throw std::runtime_error(
        "Navier-Stokes nodal field output size does not match the mesh");
  }

  const auto  vel      = space.field(0);
  const auto  pre      = space.field(1);
  const Index num_comp = vel.numComponents();
  for (Index node = 0; node < num_nodes; ++node)
  {
    ux[node] = state[vel.globalDof(node, 0)];
    uy[node] = num_comp > 1 ? state[vel.globalDof(node, 1)] : 0.0;
    uz[node] = num_comp > 2 ? state[vel.globalDof(node, 2)] : 0.0;

    pressure[node] = state[pre.globalDof(node)];
  }
}

} // namespace femx::model::ns
