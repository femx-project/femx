#include "StateFields.hpp"

#include <stdexcept>

using namespace femx;

namespace femx::model::navier
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
  for (Index in = 0; in < num_nodes; ++in)
  {
    ux[in] = state[vel.globalDof(in, 0)];
    uy[in] = num_comp > 1 ? state[vel.globalDof(in, 1)] : 0.0;
    uz[in] = num_comp > 2 ? state[vel.globalDof(in, 2)] : 0.0;

    pressure[in] = state[pre.globalDof(in)];
  }
}

} // namespace femx::model::navier
