#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/**
 * @brief Immutable compressed-sparse-row graph in one memory space.
 *
 * CsrGraph owns only the row offsets and column indices. Numeric matrix values
 * are owned separately by CsrMatrix so one graph can be reused by multiple
 * matrices with the same sparsity structure.
 */
template <MemorySpace Space>
class CsrGraph
{
public:
  /** @brief Index-vector type in this graph's memory space. */
  using IndexVector = Vector<Space, Index>;

  CsrGraph()
    : storage_(std::make_shared<Storage>())
  {
  }

  CsrGraph(const CsrGraph&)                = default;
  CsrGraph(CsrGraph&&) noexcept            = default;
  CsrGraph& operator=(const CsrGraph&)     = default;
  CsrGraph& operator=(CsrGraph&&) noexcept = default;

  template <MemorySpace S                                              = Space,
            typename std::enable_if<S == MemorySpace::Host, int>::type = 0>
  /**
   * @brief Construct and validate a host CSR graph.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param row_ptr CSR row offsets of size `rows + 1`.
   * @param col_ind CSR column indices.
   */
  CsrGraph(Index       rows,
           Index       cols,
           IndexVector row_ptr,
           IndexVector col_ind)
    : storage_(std::make_shared<Storage>(rows,
                                         cols,
                                         std::move(row_ptr),
                                         std::move(col_ind),
                                         0))
  {
    checkSizes();
    initLayoutId();
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

  template <MemorySpace S                                                = Space,
            typename std::enable_if<S == MemorySpace::Device, int>::type = 0>
  CsrGraph(Index         rows,
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
    initLayoutId();
  }

  friend void copy(const HostCsrGraph&,
                   DeviceCsrGraph&,
                   CudaContext&);

  void checkSizes() const
  {
    if (rows() < 0 || cols() < 0
        || rows() == std::numeric_limits<Index>::max())
    {
      throw std::runtime_error("CsrGraph dimensions must be non-negative");
    }
    if (rowPtr().size() != rows() + 1)
    {
      throw std::runtime_error(
          "CsrGraph row-offset size does not match its row count");
    }

    if constexpr (Space == MemorySpace::Host)
    {
      if (rowPtr()[0] != 0 || rowPtr()[rows()] != nnz())
      {
        throw std::runtime_error(
            "CsrGraph row offsets must begin at zero and end at nnz");
      }
      for (Index row = 0; row < rows(); ++row)
      {
        if (rowPtr()[row] < 0
            || rowPtr()[row] > rowPtr()[row + 1]
            || rowPtr()[row + 1] > nnz())
        {
          throw std::runtime_error(
              "CsrGraph row offsets must be monotone and in range");
        }
      }
      for (Index k = 0; k < nnz(); ++k)
      {
        if (colInd()[k] < 0 || colInd()[k] >= cols())
        {
          throw std::runtime_error(
              "CsrGraph column index is out of range");
        }
      }
    }
  }

  static std::uint64_t mix(std::uint64_t hash, std::uint64_t val) noexcept
  {
    // FNV-1a over the eight bytes of each normalized integer value.
    for (int i = 0; i < 8; ++i)
    {
      hash  ^= val & 0xffu;
      hash  *= UINT64_C(1099511628211);
      val  >>= 8;
    }
    return hash;
  }

  std::uint64_t hostLayoutId() const noexcept
  {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    hash               = mix(hash, static_cast<std::uint32_t>(rows()));
    hash               = mix(hash, static_cast<std::uint32_t>(cols()));
    hash               = mix(hash, static_cast<std::uint32_t>(nnz()));
    for (Index ptr : rowPtr())
    {
      hash = mix(hash, static_cast<std::uint32_t>(ptr));
    }
    for (Index col : colInd())
    {
      hash = mix(hash, static_cast<std::uint32_t>(col));
    }
    return hash == 0 ? 1 : hash;
  }

  void initLayoutId()
  {
    if constexpr (Space == MemorySpace::Host)
    {
      storage_->layout_id = hostLayoutId();
    }
    else if (storage_->layout_id == 0)
    {
      throw std::runtime_error(
          "Device CsrGraph must be created by copying a validated host graph");
    }
  }

  std::shared_ptr<Storage> storage_;
};

/**
 * @brief Explicitly copy a host CSR graph to device-owned storage.
 * @param src Validated host graph.
 * @param dst Device graph replaced by the copy.
 * @param ctx CUDA stream on which copies are enqueued.
 */
inline void copy(const HostCsrGraph& src,
                 DeviceCsrGraph&     dst,
                 CudaContext&        ctx)
{
  DeviceIndexVector row_ptr;
  DeviceIndexVector col_ind;
  copy(src.rowPtr(), row_ptr, ctx);
  copy(src.colInd(), col_ind, ctx);
  dst = DeviceCsrGraph(src.rows(),
                       src.cols(),
                       std::move(row_ptr),
                       std::move(col_ind),
                       src.layoutId());
}

} // namespace femx
