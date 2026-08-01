#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/Context.hpp>

namespace femx
{

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
 * CsrPattern owns only the row offsets and column indices.
 * Numeric matrix values are owned separately by CsrMatrix.
 */
template <MemorySpace Space>
class CsrPattern
{
public:
  /**
   * @brief Construct an empty zero-by-zero CSR pattern.
   */
  CsrPattern()
    : data_(std::make_shared<Data>())
  {
  }

  CsrPattern(const CsrPattern&) = default;

  CsrPattern(CsrPattern&&) noexcept = default;

  CsrPattern& operator=(const CsrPattern&) = default;

  CsrPattern& operator=(CsrPattern&&) noexcept = default;

  /**
   * @brief Construct a CSR pattern in the selected memory space.
   *
   * @param[in] rows    - Number of rows.
   * @param[in] cols    - Number of columns.
   * @param[in] row_ptr - CSR row offsets.
   * @param[in] col_ind - CSR column indices.
   * @throws std::runtime_error If validation fails.
   */
  CsrPattern(Index                rows,
             Index                cols,
             Vector<Space, Index> row_ptr,
             Vector<Space, Index> col_ind)
    : CsrPattern(rows,
                 cols,
                 std::move(row_ptr),
                 std::move(col_ind),
                 detail::newCsrLayoutId())
  {
  }

  /**
   * @brief Return the number of rows.
   */
  Index rows() const noexcept
  {
    return data_->rows;
  }

  /**
   * @brief Return the number of columns.
   */
  Index cols() const noexcept
  {
    return data_->cols;
  }

  /**
   * @brief Return the number of stored column indices.
   */
  Index nnz() const noexcept
  {
    return data_->col_ind.size();
  }

  /**
   * @brief Return the stable identity of this CSR layout.
   *
   * @return Nonzero layout identifier for a validated pattern.
   */
  std::uint64_t layoutId() const noexcept
  {
    return data_->layout_id;
  }

  /**
   * @brief Return the owned CSR row offsets.
   */
  const Vector<Space, Index>& rowPtr() const noexcept
  {
    return data_->row_ptr;
  }

  /**
   * @brief Return the owned CSR column indices.
   */
  const Vector<Space, Index>& colInd() const noexcept
  {
    return data_->col_ind;
  }

  /**
   * @brief Return the CSR row-offset data.
   */
  const Index* rowPtrData() const noexcept
  {
    return data_->row_ptr.data();
  }

  /**
   * @brief Return the CSR column-index data.
   */
  const Index* colIndData() const noexcept
  {
    return data_->col_ind.data();
  }

private:
  struct Data
  {
    Data() = default;

    Data(Index                num_rows,
         Index                num_cols,
         Vector<Space, Index> row_ptr,
         Vector<Space, Index> col_ind,
         std::uint64_t        id)
      : rows(num_rows),
        cols(num_cols),
        row_ptr(std::move(row_ptr)),
        col_ind(std::move(col_ind)),
        layout_id(id)
    {
    }

    Index                rows{0};      ///< Number of rows.
    Index                cols{0};      ///< Number of columns.
    Vector<Space, Index> row_ptr;      ///< CSR row offsets.
    Vector<Space, Index> col_ind;      ///< CSR column indices.
    std::uint64_t        layout_id{0}; ///< Stable CSR layout identifier.
  };

  CsrPattern(Index                rows,
             Index                cols,
             Vector<Space, Index> row_ptr,
             Vector<Space, Index> col_ind,
             std::uint64_t        layout_id)
    : data_(std::make_shared<Data>(rows,
                                   cols,
                                   std::move(row_ptr),
                                   std::move(col_ind),
                                   layout_id))
  {
    checkSizes();
    require(data_->layout_id != 0, "CsrPattern requires a valid layout identity");
  }

  friend void copy(const HostCsrPattern&,
                   DeviceCsrPattern&,
                   linalg::Context<MemorySpace::Device>&);

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

  std::shared_ptr<Data> data_; ///< Shared immutable pattern data.
};

/**
 * @brief Copy a Host CSR pattern to Device-owned storage.
 *
 * @param[in]  src - Source Host pattern.
 * @param[out] dst - Destination Device pattern.
 * @param[in]  ctx - Device context used to enqueue the copy.
 * @throws std::runtime_error If validation fails.
 */
inline void copy(const HostCsrPattern&                 src,
                 DeviceCsrPattern&                     dst,
                 linalg::Context<MemorySpace::Device>& ctx)
{
  DeviceVector<Index> row_ptr;
  DeviceVector<Index> col_ind;
  auto&               vec_handler = ctx.vectorHandler();

  vec_handler.copy(src.rowPtr(), row_ptr);
  vec_handler.copy(src.colInd(), col_ind);

  dst = DeviceCsrPattern(src.rows(),
                         src.cols(),
                         std::move(row_ptr),
                         std::move(col_ind),
                         src.layoutId());
}

} // namespace femx
