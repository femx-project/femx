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
 * @brief Own an immutable compressed-sparse-row pattern.
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
  using IndexVector = Vector<Space, Index>;

  /** @brief Construct an empty zero-by-zero CSR pattern. */
  CsrPattern()
    : storage_(std::make_shared<Storage>())
  {
  }

  CsrPattern(const CsrPattern&) = default;

  CsrPattern(CsrPattern&&) noexcept = default;

  CsrPattern& operator=(const CsrPattern&) = default;

  CsrPattern& operator=(CsrPattern&&) noexcept = default;

  /**
   * @brief Construct and validate a Host CSR pattern.
   *
   * @param[in] rows - Number of rows.
   * @param[in] cols - Number of columns.
   * @param[in] row_ptr - CSR row offsets.
   * @param[in] col_ind - CSR column indices.
   * @throws std::runtime_error - If dimensions or CSR indices are invalid.
   */
  template <MemorySpace S = Space, HostOnly<S> = 0>
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

  /**
   * @brief Return the number of rows.
   *
   * @return Number of rows.
   */
  Index rows() const noexcept
  {
    return storage_->rows;
  }

  /**
   * @brief Return the number of columns.
   *
   * @return Number of columns.
   */
  Index cols() const noexcept
  {
    return storage_->cols;
  }

  /**
   * @brief Return the number of stored column indices.
   *
   * @return Number of structurally nonzero entries.
   */
  Index nnz() const noexcept
  {
    return storage_->col_ind.size();
  }

  /**
   * @brief Return the stable identity of this CSR layout.
   *
   * @return Nonzero layout identifier for a validated pattern.
   */
  std::uint64_t layoutId() const noexcept
  {
    return storage_->layout_id;
  }

  /**
   * @brief Return the owned CSR row offsets.
   *
   * @return Read-only row-offset vector.
   */
  const IndexVector& rowPtr() const noexcept
  {
    return storage_->row_ptr;
  }

  /**
   * @brief Return the owned CSR column indices.
   *
   * @return Read-only column-index vector.
   */
  const IndexVector& colInd() const noexcept
  {
    return storage_->col_ind;
  }

  /**
   * @brief Return the CSR row-offset data.
   *
   * @return Read-only pointer to row offsets.
   */
  const Index* rowPtrData() const noexcept
  {
    return storage_->row_ptr.data();
  }

  /**
   * @brief Return the CSR column-index data.
   *
   * @return Read-only pointer to column indices.
   */
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

    Index         rows{0};      ///< Number of rows.
    Index         cols{0};      ///< Number of columns.
    IndexVector   row_ptr;      ///< CSR row offsets.
    IndexVector   col_ind;      ///< CSR column indices.
    std::uint64_t layout_id{0}; ///< Stable CSR layout identifier.
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

  std::shared_ptr<Storage> storage_; ///< Shared immutable pattern storage.
};

/**
 * @brief Copy a Host CSR pattern to Device-owned storage.
 *
 * @param[in] src - Source Host pattern.
 * @param[out] dst - Destination Device pattern.
 * @param[in] ctx - CUDA context used to enqueue the copy.
 * @throws std::runtime_error - If a CUDA allocation or copy fails.
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
