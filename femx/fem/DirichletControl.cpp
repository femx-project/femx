#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

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

HostCsrMatrix makeControlMatrix(
    Index                                  rows,
    Index                                  cols,
    const Array<DirichletControlMapEntry>& entries)
{
  require(cols >= 0,
          "DirichletControl received negative control parameter count");

  HostIndexVector                   row_ptr(rows + 1);
  std::set<std::pair<Index, Index>> seen;
  for (const DirichletControlMapEntry& entry : entries)
  {
    require(entry.state_row >= 0 && entry.state_row < rows,
            "DirichletControl map state row is out of range");
    require(entry.ctr_col >= 0 && entry.ctr_col < cols,
            "DirichletControl map control column is out of range");
    require(std::isfinite(entry.weight),
            "DirichletControl received non-finite map weight");
    require(seen.insert({entry.state_row, entry.ctr_col}).second,
            "DirichletControl received duplicate map entry");
    ++row_ptr[entry.state_row + 1];
  }
  for (Index row = 0; row < rows; ++row)
  {
    row_ptr[row + 1] += row_ptr[row];
  }

  HostIndexVector col_ind(entries.size());
  HostVector      vals(entries.size());
  HostIndexVector next = row_ptr;
  for (const DirichletControlMapEntry& entry : entries)
  {
    const Index k = next[entry.state_row]++;
    col_ind[k]    = entry.ctr_col;
    vals[k]       = entry.weight;
  }

  HostCsrMatrix mat(
      HostCsrPattern(rows, cols, std::move(row_ptr), std::move(col_ind)));
  mat.vals() = std::move(vals);
  return mat;
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

  require(!dof_set.empty(),
          "DirichletControl found no velocity boundary dofs");

  Array<Index> dofs;
  for (Index id : dof_set)
  {
    dofs.push_back(id);
  }

  return DirichletControl(std::move(dofs));
}

HostVector normalizedVelocityDirection(const MixedFESpace& space,
                                       const HostVector&   nrm)
{
  const Index ncomp = space.field(0).numComponents();
  require(nrm.size() == ncomp,
          "Normal velocity control direction size mismatch");

  Real norm2 = 0.0;
  for (Real val : nrm)
  {
    require(std::isfinite(val),
            "Normal velocity control direction must be finite");
    norm2 += val * val;
  }
  require(norm2 > 0.0 && std::isfinite(norm2),
          "Normal velocity control direction must be nonzero");

  HostVector dir(ncomp);
  const Real inv_norm = 1.0 / std::sqrt(norm2);
  for (Index comp = 0; comp < ncomp; ++comp)
  {
    dir[comp] = inv_norm * nrm[comp];
  }
  return dir;
}

template <typename Match>
DirichletControl makeNormalVelocityControlFromPredicate(
    const MixedFESpace& space,
    Match               match,
    const HostVector&   nrm)
{
  const auto       u_dof = space.field(0);
  const HostVector dir   = normalizedVelocityDirection(space, nrm);
  std::set<Index>  nodes;

  for (const auto& facet : space.mesh().boundaryFacets())
  {
    if (match(facet))
    {
      nodes.insert(facet.nids.begin(), facet.nids.end());
    }
  }
  require(!nodes.empty(),
          "Normal velocity control found no boundary nodes");

  Array<Index>                    state_dofs;
  Array<DirichletControlMapEntry> entries;
  state_dofs.reserve(nodes.size() * u_dof.numComponents());
  entries.reserve(nodes.size() * u_dof.numComponents());
  Index col = 0;
  for (Index node : nodes)
  {
    for (Index comp = 0; comp < u_dof.numComponents(); ++comp)
    {
      const Index row = state_dofs.size();
      state_dofs.push_back(u_dof.globalDof(node, comp));
      if (dir[comp] != 0.0)
      {
        entries.push_back({row, col, dir[comp]});
      }
    }
    ++col;
  }

  return DirichletControl(
      std::move(state_dofs), nodes.size(), std::move(entries));
}

} // namespace

DirichletControl::DirichletControl()
  : DirichletControl({}, 0, {})
{
}

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
    matrix_(makeControlMatrix(
        dofs_.size(), num_ctr_params, map_entries))
{
  std::set<Index> seen;
  for (Index id : dofs_)
  {
    require(id >= 0, "DirichletControl received negative state id");
    require(seen.insert(id).second,
            "DirichletControl received duplicate state id");
  }
}

Index DirichletControl::numStateDofs() const
{
  return dofs_.size();
}

Index DirichletControl::numControlParams() const
{
  return matrix_.cols();
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

const HostCsrMatrix& DirichletControl::matrix() const noexcept
{
  return matrix_;
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

  Array<char> used_cols(numControlParams(), 0);
  for (Index old_row = 0; old_row < numStateDofs(); ++old_row)
  {
    if (old_to_new_row[old_row] < 0)
    {
      continue;
    }
    for (Index k = matrix_.rowPtrData()[old_row];
         k < matrix_.rowPtrData()[old_row + 1];
         ++k)
    {
      used_cols[matrix_.colIndData()[k]] = 1;
    }
  }

  Array<Index> old_to_new_col(numControlParams(), -1);
  Index        num_ctr_params = 0;
  for (Index old_column = 0; old_column < numControlParams(); ++old_column)
  {
    if (used_cols[old_column] != 0)
    {
      old_to_new_col[old_column] = num_ctr_params++;
    }
  }

  Array<DirichletControlMapEntry> entries;
  entries.reserve(matrix_.nnz());
  for (Index old_row = 0; old_row < numStateDofs(); ++old_row)
  {
    const Index row = old_to_new_row[old_row];
    if (row < 0)
    {
      continue;
    }
    for (Index k = matrix_.rowPtrData()[old_row];
         k < matrix_.rowPtrData()[old_row + 1];
         ++k)
    {
      entries.push_back({row,
                         old_to_new_col[matrix_.colIndData()[k]],
                         matrix_.valsData()[k]});
    }
  }

  return DirichletControl(
      std::move(state_dofs), num_ctr_params, std::move(entries));
}

void DirichletControl::apply(const HostVector& dir,
                             HostVector&       out) const
{
  checkControlVector(dir);
  CpuContext                ctx;
  linalg::HostVectorHandler vec_handler(ctx);
  linalg::HostMatrixHandler mat_handler(ctx);
  vec_handler.resizeOrZero(out, numStateDofs());
  mat_handler.matvec(matrix_, dir.view(), out.view());
}

void DirichletControl::applyTranspose(const HostVector& dir,
                                      HostVector&       out) const
{
  checkStateVector(dir);
  CpuContext                ctx;
  linalg::HostVectorHandler vec_handler(ctx);
  linalg::HostMatrixHandler mat_handler(ctx);
  vec_handler.resizeOrZero(out, numControlParams());
  mat_handler.matvecT(matrix_, dir.view(), out.view());
}

void DirichletControl::checkDofIndex(Index i) const
{
  require(i >= 0 && i < numStateDofs(),
          "DirichletControl id index is out of range");
}

void DirichletControl::checkControlVector(
    const HostVector& ctr) const
{
  require(ctr.size() == numControlParams(),
          "DirichletControl control vector size mismatch");
}

void DirichletControl::checkStateVector(const HostVector& state) const
{
  require(state.size() == numStateDofs(),
          "DirichletControl state vector size mismatch");
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
    const HostVector&   nrm)
{
  return makeNormalVelocityControlFromPredicate(
      space,
      [ptag](const Mesh::BoundaryFacet& facet)
      {
        return facet.ptag == ptag;
      },
      nrm);
}

DirichletControl makeNormalVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname,
    const HostVector&   nrm)
{
  return makeNormalVelocityControlFromPredicate(
      space,
      [&pname](const Mesh::BoundaryFacet& facet)
      {
        return facet.pname == pname;
      },
      nrm);
}

} // namespace fem
} // namespace femx
