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
  using Graph = CsrGraph<Space>;
  using Vals  = Vector<Space>;

  explicit CsrMatrix(const Graph& graph)
    : graph_(graph), vals_(graph.nnz())
  {
  }

  template <MemorySpace S                                              = Space,
            typename std::enable_if<S == MemorySpace::Host, int>::type = 0>
  void setZero()
  {
    vals_.setZero();
  }

  template <MemorySpace S                                                = Space,
            typename std::enable_if<S == MemorySpace::Device, int>::type = 0>
  void setZero(CudaContext& ctx)
  {
    vals_.setZero(ctx);
  }

  Index rows() const noexcept
  {
    return graph_.rows();
  }

  Index cols() const noexcept
  {
    return graph_.cols();
  }

  Index nnz() const noexcept
  {
    return graph_.nnz();
  }

  const Graph& graph() const noexcept
  {
    return graph_;
  }

  const Index* rowPtrData() const noexcept
  {
    return graph_.rowPtrData();
  }

  const Index* colIndData() const noexcept
  {
    return graph_.colIndData();
  }

  Real* valsData() noexcept
  {
    return vals_.data();
  }

  const Real* valsData() const noexcept
  {
    return vals_.data();
  }

  Vals& vals() noexcept
  {
    return vals_;
  }

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
 */
inline void copy(const HostCsrMatrix& src,
                 DeviceCsrMatrix&     dst,
                 CudaContext&         ctx)
{
  detail::checkCsrMatCopy(src, dst);
  copy(src.vals(), dst.vals(), ctx);
}

/**
 * @brief Explicitly copy device matrix values to a host matrix.
 *
 * The destination must already retain the host counterpart of the source
 * graph. Synchronize context before consuming the copied host values.
 */
inline void copy(const DeviceCsrMatrix& src,
                 HostCsrMatrix&         dst,
                 CudaContext&           ctx)
{
  detail::checkCsrMatCopy(src, dst);
  copy(src.vals(), dst.vals(), ctx);
}

} // namespace femx
