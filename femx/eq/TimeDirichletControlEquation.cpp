#include <femx/eq/TimeDirichletControlEquation.hpp>

#include <stdexcept>
#include <utility>

#include <femx/linalg/SparseMatrix.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#if defined(FEMX_HAS_PETSC)
#include <femx/system/petsc/PETScSystemMatrix.hpp>
#endif

using namespace femx::system;

namespace femx
{
namespace eq
{

namespace
{

void replaceSparseRow(SparseSystemMatrix& mat,
                      Index               row,
                      Real                diag)
{
  SparseMatrix& sparse = mat.matrix();
  if (row < 0 || row >= sparse.rows())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation sparse row is out of range");
  }

  const Index* row_ptr = sparse.rowPtrData();
  const Index* col_ind = sparse.colIndData();
  Real*        values  = sparse.valuesData();

  bool has_diag = false;
  for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
  {
    values[k] = 0.0;
    if (col_ind[k] == row)
    {
      values[k] = diag;
      has_diag  = true;
    }
  }

  if (diag != 0.0 && !has_diag)
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation sparse pattern lacks diagonal");
  }
}

} // namespace

TimeDirichletControlEquation::TimeDirichletControlEquation(
    const TimeMatrixResidualEquation& base_eq,
    DirichletControl                  control,
    Vector<Index>                     fixed_dofs,
    Index                             control_param_offset,
    Index                             num_params,
    Vector<Real>                      fixed_values)
  : base_eq_(base_eq),
    ctr_(std::move(control)),
    fixed_dofs_(std::move(fixed_dofs)),
    fixed_values_(std::move(fixed_values)),
    base_prm_(base_eq.numParams()),
    control_param_offset_(control_param_offset),
    num_params_(num_params < 0
                    ? control_param_offset
                          + ctr_.numParams(base_eq.numSteps())
                    : num_params)
{
  if (base_eq_.numRes() != base_eq_.numStates())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation requires square state residuals");
  }
  if (base_eq_.numParams() != 0)
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation requires a parameter-free base equation");
  }
  if (control_param_offset_ < 0)
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation received negative control parameter offset");
  }
  if (num_params_ < control_param_offset_ + ctr_.numParams(numSteps()))
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation parameter count is too small");
  }
  if (fixed_values_.empty())
  {
    fixed_values_.resize(fixed_dofs_.size());
  }
  else if (fixed_values_.size() != fixed_dofs_.size()
           && fixed_values_.size() != numSteps() * fixed_dofs_.size())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation fixed value size mismatch");
  }
  for (Index dof : ctr_.stateDofs())
  {
    if (dof < 0 || dof >= base_eq_.numStates())
    {
      throw std::runtime_error(
          "TimeDirichletControlEquation state dof is out of range");
    }
  }
  for (Index dof : fixed_dofs_)
  {
    if (dof < 0 || dof >= base_eq_.numStates())
    {
      throw std::runtime_error(
        "TimeDirichletControlEquation fixed dof is out of range");
    }
    for (Index ctr_dof : ctr_.stateDofs())
    {
      if (dof == ctr_dof)
      {
        throw std::runtime_error(
            "TimeDirichletControlEquation received overlapping dofs");
      }
    }
  }
}

Index TimeDirichletControlEquation::numSteps() const
{
  return base_eq_.numSteps();
}

Index TimeDirichletControlEquation::numStates() const
{
  return base_eq_.numStates();
}

Index TimeDirichletControlEquation::numParams() const
{
  return num_params_;
}

Index TimeDirichletControlEquation::numRes() const
{
  return base_eq_.numRes();
}

void TimeDirichletControlEquation::res(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    Vector<Real>&       out) const
{
  checkSizes(step, x_next, x, prm);
  base_eq_.res(step, x_next, x, base_prm_, out);

  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Index row = ctr_.stateDof(i);
    const Index col = controlParamIndex(step, i);
    out[row]        = x_next[row] - prm[col];
  }
  for (Index i = 0; i < fixed_dofs_.size(); ++i)
  {
    const Index row = fixed_dofs_[i];
    out[row]        = x_next[row] - fixedValue(step, i);
  }
}

void TimeDirichletControlEquation::assembleNextStateJac(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    SystemMatrix&       out) const
{
  checkSizes(step, x_next, x, prm);
  base_eq_.assembleNextStateJac(step, x_next, x, base_prm_, out);
  replaceStateRows(out, 1.0);
}

void TimeDirichletControlEquation::assemblePrevStateJac(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    SystemMatrix&       out) const
{
  checkSizes(step, x_next, x, prm);
  base_eq_.assemblePrevStateJac(step, x_next, x, base_prm_, out);
  replaceStateRows(out, 0.0);
}

void TimeDirichletControlEquation::assembleParamJac(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    SystemMatrix&       out) const
{
  checkSizes(step, x_next, x, prm);
  out.resize(numRes(), numParams());
  out.setZero();

  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Index row = ctr_.stateDof(i);
    const Index col = controlParamIndex(step, i);
    out.set(row, col, -1.0);
  }
}

void TimeDirichletControlEquation::applyParamJac(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  checkSizes(step, x_next, x, prm);
  if (dir.size() != numParams())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation parameter direction size mismatch");
  }

  if (out.size() != numRes())
  {
    out.resize(numRes());
  }
  else
  {
    out.setZero();
  }

  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Index row  = ctr_.stateDof(i);
    const Index col  = controlParamIndex(step, i);
    out[row]        -= dir[col];
  }
}

void TimeDirichletControlEquation::applyParamJacT(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& lambda,
    Vector<Real>&       out) const
{
  checkSizes(step, x_next, x, prm);
  if (lambda.size() != numRes())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation adjoint vector size mismatch");
  }

  if (out.size() != numParams())
  {
    out.resize(numParams());
  }
  else
  {
    out.setZero();
  }

  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Index row  = ctr_.stateDof(i);
    const Index col  = controlParamIndex(step, i);
    out[col]        -= lambda[row];
  }
}

const DirichletControl&
TimeDirichletControlEquation::control() const
{
  return ctr_;
}

void TimeDirichletControlEquation::checkSizes(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm) const
{
  if (step < 0 || step >= numSteps())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation step is out of range");
  }
  if (x_next.size() != numStates()
      || x.size() != numStates()
      || prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation size mismatch");
  }
}

void TimeDirichletControlEquation::replaceStateRows(
    SystemMatrix& out,
    Real          diag) const
{
  if (out.numRows() != numRes() || out.numCols() != numStates())
  {
    throw std::runtime_error(
        "TimeDirichletControlEquation matrix size mismatch");
  }

  if (auto* sparse = dynamic_cast<SparseSystemMatrix*>(&out))
  {
    for (Index row : ctr_.stateDofs())
    {
      replaceSparseRow(*sparse, row, diag);
    }
    for (Index row : fixed_dofs_)
    {
      replaceSparseRow(*sparse, row, diag);
    }
    return;
  }

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<PETScSystemMatrix*>(&out))
  {
    Vector<Index> rows;
    rows.reserve(ctr_.numDofs() + fixed_dofs_.size());
    for (Index row : ctr_.stateDofs())
    {
      rows.push_back(row);
    }
    for (Index row : fixed_dofs_)
    {
      rows.push_back(row);
    }

    out.finalize();
    petsc->zeroRows(rows, diag);
    return;
  }
#endif

  for (Index row : ctr_.stateDofs())
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
  for (Index row : fixed_dofs_)
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

Index TimeDirichletControlEquation::controlParamIndex(Index step,
                                                      Index i) const
{
  return control_param_offset_ + ctr_.paramIndex(step, i);
}

Real TimeDirichletControlEquation::fixedValue(Index step,
                                              Index i) const
{
  if (fixed_values_.size() == fixed_dofs_.size())
  {
    return fixed_values_[i];
  }
  return fixed_values_[step * fixed_dofs_.size() + i];
}

} // namespace eq
} // namespace femx
