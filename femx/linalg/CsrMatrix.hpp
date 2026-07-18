#pragma once

#include <stdexcept>
#include <type_traits>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/**
 * @brief Numeric values for an immutable CSR graph in one memory space.
 *
 * A matrix retains an immutable shared graph handle, allowing multiple
 * matrices and maps to reuse one sparse structure without lifetime coupling.
 * Numeric values are owned exclusively by this matrix in Space.
 */
template <MemorySpace Space>
class CsrMatrix
{
public:
  /** @brief Sparsity graph type in this matrix's memory space. */
  using Graph = CsrGraph<Space>;
  /** @brief Numeric value-vector type in this matrix's memory space. */
  using Vals  = Vector<Space>;

  /** @brief Allocate zeroed numeric values for an immutable CSR graph. */
  explicit CsrMatrix(const Graph& graph)
    : graph_(graph), vals_(graph.nnz())
  {
  }

  template <MemorySpace S                                              = Space,
            typename std::enable_if<S == MemorySpace::Host, int>::type = 0>
  /** @brief Set all host matrix values to zero. */
  void setZero()
  {
    vals_.setZero();
  }

  template <MemorySpace S                                                = Space,
            typename std::enable_if<S == MemorySpace::Device, int>::type = 0>
  /** @brief Enqueue zeroing of all device matrix values on `ctx`. */
  void setZero(CudaContext& ctx)
  {
    vals_.setZero(ctx);
  }

  /** @brief Return the number of rows. */
  Index rows() const noexcept
  {
    return graph_.rows();
  }

  /** @brief Return the number of columns. */
  Index cols() const noexcept
  {
    return graph_.cols();
  }

  /** @brief Return the number of stored values. */
  Index nnz() const noexcept
  {
    return graph_.nnz();
  }

  /** @brief Return the immutable sparsity graph retained by this matrix. */
  const Graph& graph() const noexcept
  {
    return graph_;
  }

  /** @brief Return the graph's row-offset pointer in `Space`. */
  const Index* rowPtrData() const noexcept
  {
    return graph_.rowPtrData();
  }

  /** @brief Return the graph's column-index pointer in `Space`. */
  const Index* colIndData() const noexcept
  {
    return graph_.colIndData();
  }

  /** @brief Return the mutable numeric-value pointer in `Space`. */
  Real* valsData() noexcept
  {
    return vals_.data();
  }

  /** @brief Return the numeric-value pointer in `Space`. */
  const Real* valsData() const noexcept
  {
    return vals_.data();
  }

  /** @brief Return the owned mutable numeric-value vector. */
  Vals& vals() noexcept
  {
    return vals_;
  }

  /** @brief Return the owned numeric-value vector. */
  const Vals& vals() const noexcept
  {
    return vals_;
  }

private:
  Graph graph_;
  Vals  vals_;
};

namespace detail
{
template <MemorySpace SrcSpace, MemorySpace DstSpace>
inline void checkCsrMatCopy(const CsrMatrix<SrcSpace>& src,
                            const CsrMatrix<DstSpace>& dst)
{
  if (src.rows() != dst.rows() || src.cols() != dst.cols()
      || src.nnz() != dst.nnz()
      || src.graph().layoutId() != dst.graph().layoutId()
      || src.vals().size() != src.nnz()
      || dst.vals().size() != dst.nnz())
  {
    throw std::runtime_error(
        "CsrMatrix copy requires compatible source and destination graphs");
  }
}
} // namespace detail

/**
 * @brief Explicitly copy host matrix values to a device matrix.
 *
 * The destination must already retain the device counterpart of the source
 * graph.
 * @param src Host matrix whose values are copied.
 * @param dst Compatible device matrix receiving the values.
 * @param ctx CUDA stream on which the copy is enqueued.
 */
inline void copy(const CsrMatrix<MemorySpace::Host>& src,
                 CsrMatrix<MemorySpace::Device>&     dst,
                 CudaContext&                        ctx)
{
  detail::checkCsrMatCopy(src, dst);
  copy(src.vals(), dst.vals(), ctx);
}

/**
 * @brief Explicitly copy device matrix values to a host matrix.
 *
 * The destination must already retain the host counterpart of the source
 * graph. Synchronize context before consuming the copied host values.
 * @param src Device matrix whose values are copied.
 * @param dst Compatible host matrix receiving the values.
 * @param ctx CUDA stream on which the copy is enqueued.
 */
inline void copy(const CsrMatrix<MemorySpace::Device>& src,
                 CsrMatrix<MemorySpace::Host>&         dst,
                 CudaContext&                          ctx)
{
  detail::checkCsrMatCopy(src, dst);
  copy(src.vals(), dst.vals(), ctx);
}

} // namespace femx
