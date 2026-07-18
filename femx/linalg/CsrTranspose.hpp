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
  /** @brief Build the transpose graph and value permutation once. */
  explicit CsrTransposeMap(const HostCsrGraph& src_graph);

  /** @brief Return the retained source graph. */
  const HostCsrGraph& srcGraph() const noexcept
  {
    return src_graph_;
  }

  /** @brief Return the immutable transposed graph. */
  const HostCsrGraph& trGraph() const noexcept
  {
    return tr_graph_;
  }

  /** @brief Map each source value index to its transpose value index. */
  const HostIndexVector& srcToTr() const noexcept
  {
    return src_to_tr_;
  }

private:
  HostCsrGraph    src_graph_;
  HostCsrGraph    tr_graph_;
  HostIndexVector src_to_tr_;
};

/**
 * @brief Transpose host CSR values using a prebuilt map without allocation.
 * @param src Source matrix matching `map.srcGraph()`.
 * @param map Persistent transpose mapping.
 * @param dst Destination matrix matching `map.trGraph()`.
 */
void trVals(const HostCsrMatrix&       src,
            const HostCsrTransposeMap& map,
            HostCsrMatrix&             dst);

} // namespace femx
