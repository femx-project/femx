#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/**
 * @brief Persistent host CSR-transpose structure and value permutation.
 *
 * The map retains a shared handle to the immutable source graph. The
 * transposed graph and source-entry-to-transpose-entry permutation are built
 * once and owned by the map.
 */
template <>
class CsrTransposeMap<MemorySpace::Host>
{
public:
  explicit CsrTransposeMap(const HostCsrGraph& src_graph);

  const HostCsrGraph& srcGraph() const noexcept
  {
    return src_graph_;
  }

  const HostCsrGraph& trGraph() const noexcept
  {
    return tr_graph_;
  }

  const HostIndexVector& srcToTr() const noexcept
  {
    return src_to_tr_;
  }

private:
  HostCsrGraph    src_graph_;
  HostCsrGraph    tr_graph_;
  HostIndexVector src_to_tr_;
};

/** @brief Transpose host CSR values using a prebuilt map without allocation. */
void trVals(const HostCsrMatrix&       src,
            const HostCsrTransposeMap& map,
            HostCsrMatrix&             dst);

} // namespace femx
