#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>

namespace femx
{
namespace fem
{

namespace
{

HostVector<DirichletControlMapEntry> identityEntries(Index size)
{
  HostVector<DirichletControlMapEntry> entries;
  entries.reserve(size);
  for (Index i = 0; i < size; ++i)
  {
    entries.push_back({i, i, 1.0});
  }
  return entries;
}

HostCsrMatrix makeControlMatrix(
    Index                                       rows,
    Index                                       cols,
    const HostVector<DirichletControlMapEntry>& entries)
{
  require(cols >= 0,
          "DirichletControl received negative control parameter count");

  HostVector<Index>                 row_ptr(rows + 1);
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

  HostVector<Index> col_ind(entries.size());
  HostVector<Real>  vals(entries.size());
  HostVector<Index> next = row_ptr;
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
      for (Index ic = 0; ic < u_dof.numComponents(); ++ic)
      {
        dof_set.insert(u_dof.globalDof(in, ic));
      }
    }
  }

  require(!dof_set.empty(),
          "DirichletControl found no velocity boundary dofs");

  HostVector<Index> dofs;
  for (Index dof : dof_set)
  {
    dofs.push_back(dof);
  }

  return DirichletControl(std::move(dofs));
}

HostVector<Real> normalizedVelocityDirection(const MixedFESpace&     space,
                                             const HostVector<Real>& nrm)
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

  HostVector<Real> dir(ncomp);
  const Real       inv_norm = 1.0 / std::sqrt(norm2);
  for (Index ic = 0; ic < ncomp; ++ic)
  {
    dir[ic] = inv_norm * nrm[ic];
  }
  return dir;
}

template <typename Match>
DirichletControl makeNormalVelocityControlFromPredicate(
    const MixedFESpace&     space,
    Match                   match,
    const HostVector<Real>& nrm)
{
  const auto             u_dof = space.field(0);
  const HostVector<Real> dir   = normalizedVelocityDirection(space, nrm);
  std::set<Index>        nodes;

  for (const auto& facet : space.mesh().boundaryFacets())
  {
    if (match(facet))
    {
      nodes.insert(facet.nids.begin(), facet.nids.end());
    }
  }
  require(!nodes.empty(),
          "Normal velocity control found no boundary nodes");

  HostVector<Index>                    state_dofs;
  HostVector<DirichletControlMapEntry> entries;
  state_dofs.reserve(nodes.size() * u_dof.numComponents());
  entries.reserve(nodes.size() * u_dof.numComponents());
  Index col = 0;
  for (Index in : nodes)
  {
    for (Index ic = 0; ic < u_dof.numComponents(); ++ic)
    {
      const Index row = state_dofs.size();
      state_dofs.push_back(u_dof.globalDof(in, ic));
      if (dir[ic] != 0.0)
      {
        entries.push_back({row, col, dir[ic]});
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

DirichletControl::DirichletControl(HostVector<Index> dofs)
  : DirichletControl(dofs,
                     dofs.size(),
                     identityEntries(dofs.size()))
{
}

DirichletControl::DirichletControl(
    HostVector<Index>                    state_dofs,
    Index                                num_ctr_params,
    HostVector<DirichletControlMapEntry> map_entries)
  : dofs_(std::move(state_dofs)),
    matrix_(makeControlMatrix(
        dofs_.size(), num_ctr_params, map_entries))
{
  std::set<Index> seen;
  for (Index dof : dofs_)
  {
    require(dof >= 0, "DirichletControl received negative state id");
    require(seen.insert(dof).second,
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

const HostVector<Index>& DirichletControl::stateDofs() const
{
  return dofs_;
}

const HostCsrMatrix& DirichletControl::matrix() const noexcept
{
  return matrix_;
}

DirichletControl DirichletControl::withoutStateDofs(
    const HostVector<Index>& excluded) const
{
  std::set<Index>   excluded_set(excluded.begin(), excluded.end());
  HostVector<Index> old_to_new_row(numStateDofs(), -1);
  HostVector<Index> state_dofs;
  state_dofs.reserve(numStateDofs());
  for (Index old_row = 0; old_row < numStateDofs(); ++old_row)
  {
    if (excluded_set.find(dofs_[old_row]) == excluded_set.end())
    {
      old_to_new_row[old_row] = state_dofs.size();
      state_dofs.push_back(dofs_[old_row]);
    }
  }

  HostVector<char> used_cols(numControlParams(), 0);
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

  HostVector<Index> old_to_new_col(numControlParams(), -1);
  Index             num_ctr_params = 0;
  for (Index old_column = 0; old_column < numControlParams(); ++old_column)
  {
    if (used_cols[old_column] != 0)
    {
      old_to_new_col[old_column] = num_ctr_params++;
    }
  }

  HostVector<DirichletControlMapEntry> entries;
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

void DirichletControl::apply(const HostVector<Real>& dir,
                             HostVector<Real>&       out) const
{
  checkControlVector(dir);
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jacobian(ctx);
  vec_handler.resizeOrZero(out, numStateDofs());
  jacobian.apply(matrix_, dir.view(), out.view());
}

void DirichletControl::applyT(const HostVector<Real>& dir,
                              HostVector<Real>&       out) const
{
  checkStateVector(dir);
  linalg::HostContext      ctx;
  auto&                    vec_handler = ctx.vectorHandler();
  linalg::HostSystemMatrix jacobian(ctx);
  vec_handler.resizeOrZero(out, numControlParams());
  jacobian.applyT(matrix_, dir.view(), out.view());
}

void DirichletControl::checkDofIndex(Index i) const
{
  require(i >= 0 && i < numStateDofs(),
          "DirichletControl id index is out of range");
}

void DirichletControl::checkControlVector(
    const HostVector<Real>& ctr) const
{
  require(ctr.size() == numControlParams(),
          "DirichletControl control vector size mismatch");
}

void DirichletControl::checkStateVector(const HostVector<Real>& state) const
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
    const MixedFESpace&     space,
    Index                   ptag,
    const HostVector<Real>& nrm)
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
    const MixedFESpace&     space,
    const std::string&      pname,
    const HostVector<Real>& nrm)
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
