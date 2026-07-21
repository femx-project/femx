#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

namespace linalg
{
struct CudaCsrBackend;

template <class Backend>
class MatrixHandler;
} // namespace linalg

namespace detail
{
inline std::uint64_t newCsrLayoutId() noexcept
{
  static std::atomic<std::uint64_t> next_id{1};
  return next_id++;
}
} // namespace detail

/**
 * @brief Immutable compressed-sparse-row pattern in one memory space.
 *
 * CsrPattern owns only the row offsets and column indices. Numeric matrix values
 * are owned separately by CsrMatrix so one pattern can be reused by multiple
 * matrices with the same sparsity structure.
 */
template <MemorySpace Space>
class CsrPattern
{
  template <MemorySpace S>
  using HostOnly = std::enable_if_t<S == MemorySpace::Host, int>;

  template <MemorySpace S>
  using DeviceOnly = std::enable_if_t<S == MemorySpace::Device, int>;

public:
  /** @brief Index-vector type in this pattern's memory space. */
  using IndexVector = Vector<Space, Index>;

  CsrPattern()
    : storage_(std::make_shared<Storage>())
  {
  }

  CsrPattern(const CsrPattern&)                = default;
  CsrPattern(CsrPattern&&) noexcept            = default;
  CsrPattern& operator=(const CsrPattern&)     = default;
  CsrPattern& operator=(CsrPattern&&) noexcept = default;

  template <MemorySpace S = Space, HostOnly<S> = 0>
  /**
   * @brief Construct and validate a host CSR pattern.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param row_ptr CSR row offsets of size `rows + 1`.
   * @param col_ind CSR column indices.
   */
  CsrPattern(Index       rows,
             Index       cols,
             IndexVector row_ptr,
             IndexVector col_ind)
    : storage_(std::make_shared<Storage>(rows,
                                         cols,
                                         std::move(row_ptr),
                                         std::move(col_ind),
                                         detail::newCsrLayoutId()))
  {
    checkSizes();
  }

  /** @brief Return the number of rows. */
  Index rows() const noexcept
  {
    return storage_->rows;
  }

  /** @brief Return the number of columns. */
  Index cols() const noexcept
  {
    return storage_->cols;
  }

  /** @brief Return the number of structural nonzeros. */
  Index nnz() const noexcept
  {
    return storage_->col_ind.size();
  }

  /** @brief Stable identifier for this exact CSR row/column layout. */
  std::uint64_t layoutId() const noexcept
  {
    return storage_->layout_id;
  }

  /** @brief Return the owned CSR row-offset vector. */
  const IndexVector& rowPtr() const noexcept
  {
    return storage_->row_ptr;
  }

  /** @brief Return the owned CSR column-index vector. */
  const IndexVector& colInd() const noexcept
  {
    return storage_->col_ind;
  }

  /** @brief Return a pointer to the CSR row offsets in `Space`. */
  const Index* rowPtrData() const noexcept
  {
    return storage_->row_ptr.data();
  }

  /** @brief Return a pointer to the CSR column indices in `Space`. */
  const Index* colIndData() const noexcept
  {
    return storage_->col_ind.data();
  }

private:
  struct Storage
  {
    Storage() = default;

    Storage(Index         num_rows,
            Index         num_cols,
            IndexVector   row_ptr,
            IndexVector   col_ind,
            std::uint64_t id)
      : rows(num_rows),
        cols(num_cols),
        row_ptr(std::move(row_ptr)),
        col_ind(std::move(col_ind)),
        layout_id(id)
    {
    }

    Index         rows{0};
    Index         cols{0};
    IndexVector   row_ptr;
    IndexVector   col_ind;
    std::uint64_t layout_id{0};
  };

  template <MemorySpace S = Space, DeviceOnly<S> = 0>
  CsrPattern(Index         rows,
             Index         cols,
             IndexVector   row_ptr,
             IndexVector   col_ind,
             std::uint64_t layout_id)
    : storage_(std::make_shared<Storage>(rows,
                                         cols,
                                         std::move(row_ptr),
                                         std::move(col_ind),
                                         layout_id))
  {
    checkSizes();
    require(storage_->layout_id != 0,
            "Device CsrPattern requires a valid layout identity");
  }

  friend void copy(const HostCsrPattern&,
                   DeviceCsrPattern&,
                   CudaContext&);
  friend class linalg::MatrixHandler<linalg::CudaCsrBackend>;

  void checkSizes() const
  {
    require(rows() >= 0 && cols() >= 0
                && rows() != std::numeric_limits<Index>::max(),
            "CsrPattern dimensions must be non-negative");
    require(rowPtr().size() == rows() + 1,
            "CsrPattern row-offset size does not match its row count");

    if constexpr (Space == MemorySpace::Host)
    {
      require(rowPtr()[0] == 0 && rowPtr()[rows()] == nnz(),
              "CsrPattern row offsets must begin at zero and end at nnz");
      for (Index row = 0; row < rows(); ++row)
      {
        require(rowPtr()[row] >= 0
                    && rowPtr()[row] <= rowPtr()[row + 1]
                    && rowPtr()[row + 1] <= nnz(),
                "CsrPattern row offsets must be monotone and in range");
      }
      for (Index k = 0; k < nnz(); ++k)
      {
        require(colInd()[k] >= 0 && colInd()[k] < cols(),
                "CsrPattern column index is out of range");
      }
    }
  }

  std::shared_ptr<Storage> storage_;
};

/**
 * @brief Explicitly copy a host CSR pattern to device-owned storage.
 * @param src Validated host pattern.
 * @param dst Device pattern replaced by the copy.
 * @param ctx CUDA stream on which copies are enqueued.
 */
inline void copy(const HostCsrPattern& src,
                 DeviceCsrPattern&     dst,
                 CudaContext&          ctx)
{
  DeviceIndexVector row_ptr;
  DeviceIndexVector col_ind;
  row_ptr.resize(src.rowPtr().size());
  col_ind.resize(src.colInd().size());
  if (!row_ptr.empty())
  {
    cuda::copy(row_ptr.data(),
               MemorySpace::Device,
               src.rowPtrData(),
               MemorySpace::Host,
               static_cast<std::size_t>(row_ptr.size()) * sizeof(Index),
               ctx.stream());
  }
  if (!col_ind.empty())
  {
    cuda::copy(col_ind.data(),
               MemorySpace::Device,
               src.colIndData(),
               MemorySpace::Host,
               static_cast<std::size_t>(col_ind.size()) * sizeof(Index),
               ctx.stream());
  }
  dst = DeviceCsrPattern(src.rows(),
                         src.cols(),
                         std::move(row_ptr),
                         std::move(col_ind),
                         src.layoutId());
}

} // namespace femx
