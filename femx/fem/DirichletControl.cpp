#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx
{
namespace fem
{

namespace
{

Array<DirichletControlMapEntry> identityEntries(Index size)
{
  Array<DirichletControlMapEntry> entries;
  entries.reserve(size);
  for (Index i = 0; i < size; ++i)
  {
    entries.push_back({i, i, 1.0});
  }
  return entries;
}

template <typename Match>
DirichletControl makeVelocityControlFromPredicate(
    const MixedFESpace& space,
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

    for (Index in : facet.nids)
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

  Array<Index> dofs;
  for (Index id : dof_set)
  {
    dofs.push_back(id);
  }

  return DirichletControl(std::move(dofs));
}

HostVector normalizedVelocityDirection(const MixedFESpace& space,
                                       const HostVector&   normal)
{
  const Index components = space.field(0).numComponents();
  if (normal.size() != components)
  {
    throw std::runtime_error(
        "Normal velocity control direction size mismatch");
  }

  Real norm_squared = 0.0;
  for (Real value : normal)
  {
    if (!std::isfinite(value))
    {
      throw std::runtime_error(
          "Normal velocity control direction must be finite");
    }
    norm_squared += value * value;
  }
  if (norm_squared <= 0.0 || !std::isfinite(norm_squared))
  {
    throw std::runtime_error(
        "Normal velocity control direction must be nonzero");
  }

  HostVector direction(components);
  const Real inverse_norm = 1.0 / std::sqrt(norm_squared);
  for (Index component = 0; component < components; ++component)
  {
    direction[component] = inverse_norm * normal[component];
  }
  return direction;
}

template <typename Match>
DirichletControl makeNormalVelocityControlFromPredicate(
    const MixedFESpace& space,
    Match               match,
    const HostVector&   normal)
{
  const auto       u_dof = space.field(0);
  const HostVector direction =
      normalizedVelocityDirection(space, normal);
  std::set<Index> nodes;

  for (const auto& facet : space.mesh().boundaryFacets())
  {
    if (match(facet))
    {
      nodes.insert(facet.nids.begin(), facet.nids.end());
    }
  }
  if (nodes.empty())
  {
    throw std::runtime_error(
        "Normal velocity control found no boundary nodes");
  }

  Array<Index>                    state_dofs;
  Array<DirichletControlMapEntry> entries;
  state_dofs.reserve(nodes.size() * u_dof.numComponents());
  entries.reserve(nodes.size() * u_dof.numComponents());
  Index column = 0;
  for (Index node : nodes)
  {
    for (Index component = 0; component < u_dof.numComponents(); ++component)
    {
      const Index row = state_dofs.size();
      state_dofs.push_back(u_dof.globalDof(node, component));
      if (direction[component] != 0.0)
      {
        entries.push_back({row, column, direction[component]});
      }
    }
    ++column;
  }

  return DirichletControl(
      std::move(state_dofs), nodes.size(), std::move(entries));
}

} // namespace

DirichletControl::DirichletControl(Array<Index> dofs)
  : DirichletControl(dofs,
                     dofs.size(),
                     identityEntries(dofs.size()))
{
}

DirichletControl::DirichletControl(
    Array<Index>                    state_dofs,
    Index                           num_ctr_params,
    Array<DirichletControlMapEntry> map_entries)
  : dofs_(std::move(state_dofs)),
    num_ctr_params_(num_ctr_params),
    map_entries_(std::move(map_entries))
{
  if (num_ctr_params_ < 0)
  {
    throw std::runtime_error(
        "DirichletControl received negative control parameter count");
  }

  std::set<Index> seen;
  for (Index id : dofs_)
  {
    if (id < 0)
    {
      throw std::runtime_error(
          "DirichletControl received negative state id");
    }
    if (!seen.insert(id).second)
    {
      throw std::runtime_error(
          "DirichletControl received duplicate state id");
    }
  }

  std::set<std::pair<Index, Index>> seen_entries;
  for (const DirichletControlMapEntry& entry : map_entries_)
  {
    if (entry.state_row < 0 || entry.state_row >= numStateDofs())
    {
      throw std::runtime_error(
          "DirichletControl map state row is out of range");
    }
    if (entry.ctr_col < 0
        || entry.ctr_col >= numControlParams())
    {
      throw std::runtime_error(
          "DirichletControl map control column is out of range");
    }
    if (!std::isfinite(entry.weight))
    {
      throw std::runtime_error(
          "DirichletControl received non-finite map weight");
    }
    if (!seen_entries.insert({entry.state_row, entry.ctr_col}).second)
    {
      throw std::runtime_error(
          "DirichletControl received duplicate map entry");
    }
  }
}

Index DirichletControl::numStateDofs() const
{
  return dofs_.size();
}

Index DirichletControl::numControlParams() const
{
  return num_ctr_params_;
}

Index DirichletControl::stateDof(Index i) const
{
  checkDofIndex(i);
  return dofs_[i];
}

const Array<Index>& DirichletControl::stateDofs() const
{
  return dofs_;
}

const Array<DirichletControlMapEntry>&
DirichletControl::mapEntries() const
{
  return map_entries_;
}

DirichletControl DirichletControl::withoutStateDofs(
    const Array<Index>& excluded) const
{
  std::set<Index> excluded_set(excluded.begin(), excluded.end());
  Array<Index>    old_to_new_row(numStateDofs(), -1);
  Array<Index>    state_dofs;
  state_dofs.reserve(numStateDofs());
  for (Index old_row = 0; old_row < numStateDofs(); ++old_row)
  {
    if (excluded_set.find(dofs_[old_row]) == excluded_set.end())
    {
      old_to_new_row[old_row] = state_dofs.size();
      state_dofs.push_back(dofs_[old_row]);
    }
  }

  Array<char> used_columns(numControlParams(), 0);
  for (const DirichletControlMapEntry& entry : map_entries_)
  {
    if (old_to_new_row[entry.state_row] >= 0)
    {
      used_columns[entry.ctr_col] = 1;
    }
  }

  Array<Index> old_to_new_column(numControlParams(), -1);
  Index        num_ctr_params = 0;
  for (Index old_column = 0; old_column < numControlParams(); ++old_column)
  {
    if (used_columns[old_column] != 0)
    {
      old_to_new_column[old_column] = num_ctr_params++;
    }
  }

  Array<DirichletControlMapEntry> entries;
  entries.reserve(map_entries_.size());
  for (const DirichletControlMapEntry& entry : map_entries_)
  {
    const Index row = old_to_new_row[entry.state_row];
    if (row >= 0)
    {
      entries.push_back(
          {row, old_to_new_column[entry.ctr_col], entry.weight});
    }
  }

  return DirichletControl(
      std::move(state_dofs), num_ctr_params, std::move(entries));
}

void DirichletControl::apply(const HostVector& direction,
                             HostVector&       out) const
{
  checkControlVector(direction);
  resizeOrZero(out, numStateDofs());
  for (const DirichletControlMapEntry& entry : map_entries_)
  {
    out[entry.state_row] += entry.weight * direction[entry.ctr_col];
  }
}

void DirichletControl::applyTranspose(const HostVector& direction,
                                      HostVector&       out) const
{
  checkStateVector(direction);
  resizeOrZero(out, numControlParams());
  for (const DirichletControlMapEntry& entry : map_entries_)
  {
    out[entry.ctr_col] += entry.weight * direction[entry.state_row];
  }
}

void DirichletControl::checkDofIndex(Index i) const
{
  if (i < 0 || i >= numStateDofs())
  {
    throw std::runtime_error("DirichletControl id index is out of range");
  }
}

void DirichletControl::checkControlVector(
    const HostVector& control) const
{
  if (control.size() != numControlParams())
  {
    throw std::runtime_error(
        "DirichletControl control vector size mismatch");
  }
}

void DirichletControl::checkStateVector(const HostVector& state) const
{
  if (state.size() != numStateDofs())
  {
    throw std::runtime_error(
        "DirichletControl state vector size mismatch");
  }
}

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    Index               ptag)
{
  return makeVelocityControlFromPredicate(
      space,
      [ptag](const Mesh::BoundaryFacet& facet)
      {
        return facet.ptag == ptag;
      });
}

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname)
{
  return makeVelocityControlFromPredicate(
      space,
      [&pname](const Mesh::BoundaryFacet& facet)
      {
        return facet.pname == pname;
      });
}

DirichletControl makeNormalVelocityControl(
    const MixedFESpace& space,
    Index               ptag,
    const HostVector&   normal)
{
  return makeNormalVelocityControlFromPredicate(
      space,
      [ptag](const Mesh::BoundaryFacet& facet)
      {
        return facet.ptag == ptag;
      },
      normal);
}

DirichletControl makeNormalVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname,
    const HostVector&   normal)
{
  return makeNormalVelocityControlFromPredicate(
      space,
      [&pname](const Mesh::BoundaryFacet& facet)
      {
        return facet.pname == pname;
      },
      normal);
}

} // namespace fem
} // namespace femx
