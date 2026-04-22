#include <stdexcept>

#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/linalg/DenseMatrix.hpp>
#include <refem/linalg/FixedSparsityPattern.hpp>
#include <refem/linalg/LocalAssembler.hpp>
#include <refem/linalg/MatrixBackend.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

LocalAssembler::LocalAssembler(const FESpace& space,
                               AssemblyMode mode)
  : fe_space_(&space),
    mode_(mode)
{
}

LocalAssembler::LocalAssembler(const BlockFESpace& space,
                               AssemblyMode        mode)
  : block_space_(&space),
    mode_(mode)
{
}

LocalAssembler::LocalAssembler(const FESpace&              space,
                               const FixedSparsityPattern& pattern,
                               AssemblyMode                mode)
  : fe_space_(&space),
    pattern_(&pattern),
    mode_(mode)
{
}

LocalAssembler::LocalAssembler(const BlockFESpace&         space,
                               const FixedSparsityPattern& pattern,
                               AssemblyMode                mode)
  : block_space_(&space),
    pattern_(&pattern),
    mode_(mode)
{
}

void LocalAssembler::addLocalMatrix(index_type         ic,
                                    const DenseMatrix& Ke,
                                    SparseMatrix&      A) const
{
  if (pattern_ == nullptr)
  {
    throw std::runtime_error(
        "LocalAssembler::addLocalMatrix requires a sparsity pattern");
  }
  if (A.backend() != MatrixBackend::HostCsr)
  {
    throw std::runtime_error(
        "LocalAssembler::addLocalMatrix currently supports HostCsr only");
  }
  if (&A.pattern() != pattern_)
  {
    throw std::runtime_error(
        "LocalAssembler sparsity pattern does not match matrix pattern");
  }
  if (ic >= pattern_->numElems())
  {
    throw std::runtime_error("Cell index is out of range");
  }

  const index_type ndofs = pattern_->elemNumDofs(ic);
  if (Ke.rows() != ndofs || Ke.cols() != ndofs)
  {
    throw std::runtime_error("Local matrix size does not match cell dofs");
  }

  const index_type offset = pattern_->elemCooOffset(ic);
  real_type*       values = A.valuesData();

  index_type local_index = 0;
  for (index_type i = 0; i < ndofs; ++i)
  {
    for (index_type j = 0; j < ndofs; ++j)
    {
      const index_type coo_index = offset + local_index;
      const index_type csr_index = pattern_->mapToCsr(coo_index);

      if (mode_ == AssemblyMode::Atomic)
      {
#pragma omp atomic update
        values[static_cast<std::size_t>(csr_index)] += Ke(i, j);
      }
      else
      {
        values[static_cast<std::size_t>(csr_index)] += Ke(i, j);
      }

      ++local_index;
    }
  }
}

void LocalAssembler::addLocalVector(index_type    ic,
                                    const Vector& Fe,
                                    Vector&       b)
{
  elemDofs(ic);
  if (static_cast<index_type>(elem_dofs_.size()) != Fe.size())
  {
    throw std::runtime_error("Local vector size does not match cell dofs");
  }

  for (index_type i = 0; i < Fe.size(); ++i)
  {
    const index_type dof = elem_dofs_[static_cast<std::size_t>(i)];

    if (dof < 0 || dof >= b.size())
    {
      throw std::runtime_error("Vector dof is out of range");
    }

    real_type* values = b.data();
    if (mode_ == AssemblyMode::Atomic)
    {
#pragma omp atomic update
      values[static_cast<std::size_t>(dof)] += Fe[i];
    }
    else
    {
      values[static_cast<std::size_t>(dof)] += Fe[i];
    }
  }
}

void LocalAssembler::elemDofs(index_type ic)
{
  if (fe_space_ != nullptr)
  {
    fe_space_->elemDofs(ic, elem_dofs_);
    return;
  }
  if (block_space_ != nullptr)
  {
    block_space_->elemDofs(ic, elem_dofs_);
    return;
  }

  throw std::runtime_error("LocalAssembler has no finite element space");
}

} // namespace refem
