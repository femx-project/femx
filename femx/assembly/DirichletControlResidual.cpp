#include <set>
#include <stdexcept>
#include <utility>

#include <femx/assembly/DirichletControlResidual.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
#include <femx/state/Linearization.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScAssemblyMatrix.hpp>
#endif
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace assembly
{
namespace
{

void replaceSparseRow(CsrAssemblyMatrix& mat,
                      Index              row,
                      Real               diag)
{
  CsrMatrix& sparse = mat.mat();
  if (row < 0 || row >= sparse.rows())
  {
    throw std::runtime_error(
        "DirichletControlResidual sparse row is out of range");
  }

  const Index* rp   = sparse.rowPtrData();
  const Index* ci   = sparse.colIndData();
  Real*        vals = sparse.valuesData();

  bool has_diag = false;
  for (Index k = rp[row]; k < rp[row + 1]; ++k)
  {
    vals[k] = 0.0;
    if (ci[k] == row)
    {
      vals[k]  = diag;
      has_diag = true;
    }
  }
  if (diag != 0.0 && !has_diag)
  {
    throw std::runtime_error(
        "DirichletControlResidual sparse pattern lacks diagonal");
  }
}

} // namespace

DirichletControlResidual::DirichletControlResidual(
    const Residual&       base,
    fem::DirichletControl ctr,
    Vector<Index>         fdofs,
    Index                 ctr_param_offset,
    Index                 num_param,
    Vector<Real>          fvals)
  : base_(base),
    ctr_(std::move(ctr)),
    fdofs_(std::move(fdofs)),
    fvals_(std::move(fvals)),
    base_dims_(base.dims()),
    dims_(base_dims_),
    ctr_param_offset_(ctr_param_offset)
{
  if (base_dims_.num_residuals != base_dims_.num_states)
  {
    throw std::runtime_error(
        "DirichletControlResidual requires square state residuals");
  }
  if (base_dims_.num_param != 0)
  {
    throw std::runtime_error(
        "DirichletControlResidual requires a parameter-free base residual");
  }
  if (ctr_param_offset_ < 0)
  {
    throw std::runtime_error(
        "DirichletControlResidual received negative parameter offset");
  }

  const Index required_ctr_params = ctr_.numControlParams();
  dims_.num_param =
      num_param < 0
          ? ctr_param_offset_ + required_ctr_params
          : num_param;
  if (dims_.num_param < ctr_param_offset_ + required_ctr_params)
  {
    throw std::runtime_error(
        "DirichletControlResidual parameter count is too small");
  }
  base_prm_.resize(base_dims_.num_param);

  if (fvals_.empty())
  {
    fvals_.resize(fdofs_.size());
  }
  else if (fvals_.size() != fdofs_.size())
  {
    throw std::runtime_error(
        "DirichletControlResidual fixed value size mismatch");
  }

  std::set<Index> rows;
  for (Index id : ctr_.stateDofs())
  {
    if (id < 0 || id >= base_dims_.num_states)
    {
      throw std::runtime_error(
          "DirichletControlResidual control id is out of range");
    }
    if (!rows.insert(id).second)
    {
      throw std::runtime_error(
          "DirichletControlResidual received duplicate control dofs");
    }
  }
  for (Index id : fdofs_)
  {
    if (id < 0 || id >= base_dims_.num_states)
    {
      throw std::runtime_error(
          "DirichletControlResidual fixed id is out of range");
    }
    if (!rows.insert(id).second)
    {
      throw std::runtime_error(
          "DirichletControlResidual received overlapping dofs");
    }
  }
}

Dimensions DirichletControlResidual::dims() const
{
  return dims_;
}

const fem::DirichletControl& DirichletControlResidual::control() const
{
  return ctr_;
}

void DirichletControlResidual::res(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   Vector<Real>&       out) const
{
  checkVectorSizes(state, prm);

  base_.res(state, base_prm_, out);
  if (out.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "DirichletControlResidual base residual size mismatch");
  }

  const Vector<Real> values = controlValues(prm);
  for (Index i = 0; i < ctr_.numStateDofs(); ++i)
  {
    const Index row = ctr_.stateDof(i);
    out[row]        = state[row] - values[i];
  }
  for (Index i = 0; i < fdofs_.size(); ++i)
  {
    const Index row = fdofs_[i];
    out[row]        = state[row] - fixedValue(i);
  }
}

void DirichletControlResidual::linearize(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    Linearization&      out) const
{
  checkVectorSizes(state, prm);

  auto* mat_out = dynamic_cast<MatrixLinearization*>(&out);
  if (mat_out == nullptr)
  {
    throw std::runtime_error(
        "DirichletControlResidual requires MatrixLinearization output");
  }

  base_.linearize(state, base_prm_, out);
  replaceStateRows(mat_out->stateMat(), 1.0);
  mat_out->stateMat().finalize();

  assembleParamJac(mat_out->paramMat());
  mat_out->paramMat().finalize();
}

void DirichletControlResidual::checkVectorSizes(
    const Vector<Real>& state,
    const Vector<Real>& prm) const
{
  if (state.size() != dims_.num_states || prm.size() != dims_.num_param)
  {
    throw std::runtime_error(
        "DirichletControlResidual vector size mismatch");
  }
}

void DirichletControlResidual::replaceStateRows(MatrixBuilder& out,
                                                Real           diag) const
{
  if (out.numRows() != dims_.num_residuals
      || out.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "DirichletControlResidual state matrix size mismatch");
  }

  const Vector<Index> rows = constrainedRows();
  if (auto* sparse = dynamic_cast<CsrAssemblyMatrix*>(&out))
  {
    for (Index row : rows)
    {
      replaceSparseRow(*sparse, row, diag);
    }
    return;
  }

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<PETScAssemblyMatrix*>(&out))
  {
    out.finalize();
    petsc->zeroRows(rows, diag);
    return;
  }
#endif

  for (Index row : rows)
  {
    for (Index col = 0; col < out.numCols(); ++col)
    {
      out.set(row, col, 0.0);
    }
    if (diag != 0.0)
    {
      out.set(row, row, diag);
    }
  }
}

void DirichletControlResidual::assembleParamJac(MatrixBuilder& out) const
{
  out.resize(dims_.num_residuals, dims_.num_param);
  out.setZero();

  for (const fem::DirichletControlMapEntry& entry : ctr_.mapEntries())
  {
    out.set(ctr_.stateDof(entry.state_row),
            ctrIndex(entry.ctr_col),
            -entry.weight);
  }
}

Vector<Real> DirichletControlResidual::controlValues(
    const Vector<Real>& prm) const
{
  Vector<Real> control(ctr_.numControlParams());
  for (Index i = 0; i < control.size(); ++i)
  {
    control[i] = prm[ctrIndex(i)];
  }

  Vector<Real> values;
  ctr_.apply(control, values);
  return values;
}

Real DirichletControlResidual::fixedValue(Index i) const
{
  return fvals_[i];
}

Index DirichletControlResidual::ctrIndex(Index i) const
{
  return ctr_param_offset_ + i;
}

Vector<Index> DirichletControlResidual::constrainedRows() const
{
  Vector<Index> rows;
  rows.reserve(ctr_.numStateDofs() + fdofs_.size());
  for (Index row : ctr_.stateDofs())
  {
    rows.push_back(row);
  }
  for (Index row : fdofs_)
  {
    rows.push_back(row);
  }
  return rows;
}

} // namespace assembly
} // namespace femx
