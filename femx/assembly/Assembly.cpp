#include <femx/assembly/Assembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/DenseMatrix.hpp>

namespace femx::assembly
{

void addElem(const HostAssemblyMap& map,
             Index                  ie,
             const DenseMatrix&     elem_mat,
             HostCsrMatrix&         mat,
             bool                   atomic)
{
  require(ie >= 0 && ie < map.numElems(),
          "Host CSR element index is out of range");
  require(mat.pattern().layoutId() == map.pattern().layoutId(),
          "Host CSR matrix must use the AssemblyMap layout");

  const auto map_v = map.view();
  require(elem_mat.rows() == map_v.numResDofs(ie)
              && elem_mat.cols() == map_v.numStateDofs(ie),
          "Host CSR element matrix size does not match AssemblyMap");

  const Index first = map_v.jac_offsets[ie];
  const Index count = map_v.jac_offsets[ie + 1] - first;
  const Real* src   = elem_mat.data();
  Real*       dst   = mat.valsData();
  for (Index i = 0; i < count; ++i)
  {
    const Index k = map_v.jac_map[first + i];
    if (atomic)
    {
#pragma omp atomic update
      dst[k] += src[i];
    }
    else
    {
      dst[k] += src[i];
    }
  }
}

void replaceRows(HostCsrMatrix&      mat,
                 const Array<Index>& rows,
                 Real                diag)
{
  for (Index row : rows)
  {
    require(row >= 0 && row < mat.rows(),
            "Host CSR row is out of range");

    bool has_diag = false;
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      mat.valsData()[k] = 0.0;
      if (mat.colIndData()[k] == row)
      {
        mat.valsData()[k] = diag;
        has_diag          = true;
      }
    }
    require(diag == 0.0 || has_diag,
            "Host CSR pattern lacks a constrained diagonal");
  }
}

void eliminateColumns(HostCsrMatrix&      mat,
                      const Array<Index>& rows,
                      HostVector&         rhs)
{
  require(rhs.size() == mat.rows(),
          "Host CSR column elimination RHS size mismatch");

  Array<char> constrained(mat.cols(), 0);
  for (Index row : rows)
  {
    require(row >= 0 && row < mat.cols(),
            "Host CSR constrained column is out of range");
    constrained[row] = 1;
  }

  for (Index row = 0; row < mat.rows(); ++row)
  {
    if (constrained[row] != 0)
    {
      continue;
    }
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      const Index col = mat.colIndData()[k];
      if (constrained[col] != 0)
      {
        rhs[row]          -= mat.valsData()[k] * rhs[col];
        mat.valsData()[k]  = 0.0;
      }
    }
  }
}

} // namespace femx::assembly
