#pragma once

#include <mpi.h>

#include <memory>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/native/HostVectorHandler.hpp>

namespace femx::linalg
{

class PETScPartition;

/**
 * @brief Own MPI execution resources for Host assembly.
 */
class MpiContext final : public Context<MemorySpace::Host>
{
public:
  /**
   * @brief Duplicate and own an MPI communicator.
   *
   * @param[in] comm - Communicator to duplicate.
   * @throws - If MPI is unavailable or duplication fails.
   */
  explicit MpiContext(MPI_Comm comm = MPI_COMM_SELF);

  /**
   * @brief Release the owned MPI communicator.
   */
  ~MpiContext() override;

  MpiContext(const MpiContext&)            = delete;
  MpiContext& operator=(const MpiContext&) = delete;
  MpiContext(MpiContext&&)                 = delete;
  MpiContext& operator=(MpiContext&&)      = delete;

  /**
   * @brief Return the owned Host vector operations.
   */
  HostVectorHandler& vectorHandler() noexcept override;

  /**
   * @brief Return the element range that may contain this rank's work.
   *
   * A graph partition requires the full range because owned elements need not
   * be contiguous. Without a graph partition, this is the rank's contiguous
   * fallback range.
   *
   * @param[in] count - Global element count.
   * @return Rank-local half-open element range.
   * @throws - If `count` is negative or MPI fails.
   */
  IndexRange elementRange(Index count) const override;

  /**
   * @brief Report whether this rank owns an element contribution.
   *
   * Graph-partitioned rows are used when a PETSc matrix has installed a
   * partition. Otherwise the contiguous fallback element range is used.
   *
   * @param[in] element - Global element index.
   * @param[in] count - Global element count.
   * @param[in] rows - Element residual rows in application numbering.
   * @return `true` when this rank should evaluate the element.
   * @throws - If an index is invalid or MPI fails.
   */
  bool ownsElement(
      Index                       element,
      Index                       count,
      HostVectorView<const Index> rows) const override;

  /**
   * @brief Sum replicated Host values over the communicator.
   *
   * @param[in,out] vals - Values replaced by their communicator-wide sums.
   * @throws - If MPI fails.
   */
  void allReduceSum(HostVectorView<Real> vals) const override;

  /**
   * @brief Complete pending Host work; MPI operations are synchronous here.
   */
  void sync() const override;

  /**
   * @brief Return the owned MPI communicator.
   */
  MPI_Comm comm() const noexcept;

  /**
   * @brief Install the graph partition used for element ownership.
   *
   * @param[in] partition - Shared immutable PETSc graph partition.
   */
  void setPartition(
      std::shared_ptr<const PETScPartition> partition) noexcept;

private:
  MPI_Comm          comm_{MPI_COMM_NULL}; ///< Owned MPI communicator.
  int               rank_{0};             ///< Rank in the owned communicator.
  int               size_{1};             ///< Size of the owned communicator.
  HostVectorHandler vec_handler_;         ///< Owned Host vector operations.
  std::shared_ptr<const PETScPartition>
      partition_; ///< Graph partition shared with the system matrix.
};

} // namespace femx::linalg
