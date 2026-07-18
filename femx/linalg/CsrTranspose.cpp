#include <stdexcept>
#include <utility>

#include <femx/linalg/CsrTranspose.hpp>

namespace femx
{
CsrTransposeMap<MemorySpace::Host>::CsrTransposeMap(
    const HostCsrGraph& src_graph)
  : src_graph_(src_graph)
{
  HostIndexVector row_ptr(src_graph.cols() + 1, 0);
  const Index*    src_rows = src_graph.rowPtrData();
  const Index*    src_cols = src_graph.colIndData();

  for (Index k = 0; k < src_graph.nnz(); ++k)
  {
    ++row_ptr[src_cols[k] + 1];
  }
  for (Index row = 0; row < src_graph.cols(); ++row)
  {
    row_ptr[row + 1] += row_ptr[row];
  }

  HostIndexVector next = row_ptr;
  HostIndexVector cols(src_graph.nnz());
  src_to_tr_.resize(src_graph.nnz());

  for (Index row = 0; row < src_graph.rows(); ++row)
  {
    for (Index k = src_rows[row]; k < src_rows[row + 1]; ++k)
    {
      const Index tr_row = src_cols[k];
      const Index tr_k   = next[tr_row]++;
      cols[tr_k]         = row;
      src_to_tr_[k]      = tr_k;
    }
  }

  tr_graph_ = HostCsrGraph(src_graph.cols(),
                           src_graph.rows(),
                           std::move(row_ptr),
                           std::move(cols));
}

void trVals(const HostCsrMatrix&       src,
            const HostCsrTransposeMap& map,
            HostCsrMatrix&             dst)
{
  if (src.graph().layoutId() != map.srcGraph().layoutId()
      || dst.graph().layoutId() != map.trGraph().layoutId())
  {
    throw std::runtime_error("CSR transpose graph mismatch");
  }

  const Real*  src_vals  = src.valsData();
  Real*        tr_vals   = dst.valsData();
  const Index* src_to_tr = map.srcToTr().data();
  for (Index k = 0; k < src.nnz(); ++k)
  {
    tr_vals[src_to_tr[k]] = src_vals[k];
  }
}

} // namespace femx
