#include <femx/bc/DirichletControl.hpp>

#include <set>
#include <stdexcept>
#include <utility>

#include <femx/mesh/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx
{

namespace
{

template <typename Match>
DirichletControl buildVelocityControl(const MixedFESpace& space,
                                      Match               match)
{
  const auto      u_dof = space.field(0);
  std::set<Index> dof_set;

  for (const auto& facet : space.mesh().boundaryFacets())
  {
    if (!match(facet))
    {
      continue;
    }

    for (Index in : facet.node_ids)
    {
      for (Index d = 0; d < u_dof.numComponents(); ++d)
      {
        dof_set.insert(u_dof.globalDof(in, d));
      }
    }
  }

  if (dof_set.empty())
  {
    throw std::runtime_error(
        "DirichletControl found no velocity boundary dofs");
  }

  Vector<Index> dofs;
  dofs.reserve(static_cast<Index>(dof_set.size()));
  for (Index dof : dof_set)
  {
    dofs.push_back(dof);
  }

  return DirichletControl(std::move(dofs));
}

} // namespace

DirichletControl::DirichletControl(Vector<Index> dofs)
  : dofs_(std::move(dofs))
{
  std::set<Index> seen;
  for (Index dof : dofs_)
  {
    if (dof < 0)
    {
      throw std::runtime_error(
          "DirichletControl received negative state dof");
    }
    if (!seen.insert(dof).second)
    {
      throw std::runtime_error(
          "DirichletControl received duplicate state dof");
    }
  }
}

Index DirichletControl::numDofs() const
{
  return dofs_.size();
}

Index DirichletControl::numParams(Index steps) const
{
  return steps * numDofs();
}

Index DirichletControl::stateDof(Index i) const
{
  checkDofIndex(i);
  return dofs_[i];
}

Index DirichletControl::paramIndex(Index step, Index i) const
{
  checkDofIndex(i);
  return step * numDofs() + i;
}

const Vector<Index>& DirichletControl::stateDofs() const
{
  return dofs_;
}

void DirichletControl::checkDofIndex(Index i) const
{
  if (i < 0 || i >= numDofs())
  {
    throw std::runtime_error("DirichletControl dof index is out of range");
  }
}

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    Index               physical_tag)
{
  return buildVelocityControl(
      space,
      [physical_tag](const Mesh::BoundaryFacet& facet)
      {
        return facet.physical_tag == physical_tag;
      });
}

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    const std::string&  physical_name)
{
  return buildVelocityControl(
      space,
      [&physical_name](const Mesh::BoundaryFacet& facet)
      {
        return facet.physical_name == physical_name;
      });
}

} // namespace femx
