#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/** @brief Persistent CSR-transpose map stored on Device. */
using DeviceCsrTransposeMap = CsrTransposeMap<MemorySpace::Device>;

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
 * @brief Device counterpart of a persistent host CSR-transpose map.
 *
 * The map retains the already-copied source graph and owns one Device copy of
 * the transposed graph and value permutation.
 */
template <>
class CsrTransposeMap<MemorySpace::Device>
{
public:
  /** @brief Construct an empty map to be initialized with `copy()`. */
  CsrTransposeMap() = default;

  /** @brief Return the retained Device source graph. */
  const DeviceCsrGraph& srcGraph() const noexcept
  {
    return src_graph_;
  }

  /** @brief Return the immutable Device transpose graph. */
  const DeviceCsrGraph& trGraph() const noexcept
  {
    return tr_graph_;
  }

  /** @brief Map each Device source value index to its transpose index. */
  const DeviceIndexVector& srcToTr() const noexcept
  {
    return src_to_tr_;
  }

private:
  friend void copy(const HostCsrTransposeMap&,
                   const DeviceCsrGraph&,
                   DeviceCsrTransposeMap&,
                   CudaContext&);

  DeviceCsrGraph    src_graph_;
  DeviceCsrGraph    tr_graph_;
  DeviceIndexVector src_to_tr_;
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

/**
 * @brief Initialize a Device transpose map from a persistent host map.
 *
 * `src_graph` must be the Device copy of `src.srcGraph()`. It is retained by
 * shared handle rather than copied again. The transpose graph and permutation
 * are allocated and copied once while initializing `dst`.
 * @param src Persistent host transpose map.
 * @param src_graph Existing Device source graph.
 * @param dst Device transpose map replaced by the initialized copy.
 * @param ctx CUDA stream on which the copies are enqueued.
 */
void copy(const HostCsrTransposeMap& src,
          const DeviceCsrGraph&      src_graph,
          DeviceCsrTransposeMap&     dst,
          CudaContext&               ctx);

/**
 * @brief Update Device transpose values through a prebuilt permutation.
 *
 * No allocation or topology copy is performed.
 * @param src Device source matrix matching `map.srcGraph()`.
 * @param map Persistent Device transpose mapping.
 * @param dst Device destination matrix matching `map.trGraph()`.
 * @param ctx CUDA stream on which the update is enqueued.
 */
void trVals(const DeviceCsrMatrix&       src,
            const DeviceCsrTransposeMap& map,
            DeviceCsrMatrix&             dst,
            CudaContext&                 ctx);

} // namespace femx
