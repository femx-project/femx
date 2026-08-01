#pragma once

#include <mpi.h>

#include <memory>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/host/HostMatrixHandler.hpp>
#include <femx/linalg/host/HostVectorHandler.hpp>

namespace femx::linalg
{

class PETScPartition;

/**
 * @brief Own MPI execution resources for Host assembly.
 */
class MpiContext final : public Context<MemorySpace::Host>
{
  using Base = Context<MemorySpace::Host>;

public:
  /**
   * @brief Duplicate and own an MPI communicator.
   *
   * @param[in] comm - Communicator to duplicate.
   * @throws std::runtime_error If validation fails.
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
   * @copydoc Base::vectorHandler()
   */
  VectorHandler<MemorySpace::Host>& vectorHandler() noexcept override;

  /**
   * @copydoc Base::matrixHandler()
   *
   * @details Returns operations for local Host matrices.
   */
  MatrixHandler<MemorySpace::Host>& matrixHandler() noexcept override;

  /**
   * @copydoc Base::elementRange()
   *
   * A graph partition requires the full range because owned elements need not
   * be contiguous. Without a graph partition, this is the rank's contiguous
   * fallback range.
   */
  IndexRange elementRange(Index count) const override;

  /**
   * @copydoc Base::ownsElement()
   *
   * Graph-partitioned rows are used when a PETSc matrix has installed a
   * partition. Otherwise the contiguous fallback element range is used.
   */
  bool ownsElement(
      Index                       element,
      Index                       count,
      HostVectorView<const Index> rows) const override;

  /**
   * @copydoc Base::allReduceSum()
   *
   * @details Reduces values over the owned communicator.
   */
  void allReduceSum(HostVectorView<Real> vals) const override;

  /**
   * @copydoc Base::sync()
   *
   * @details MPI operations are synchronous here.
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
  HostMatrixHandler mat_handler_;         ///< Owned local Host matrix operations.
  HostVectorHandler vec_handler_;         ///< Owned Host vector operations.

  std::shared_ptr<const PETScPartition> partition_; ///< Graph partition shared with the system matrix.
};

} // namespace femx::linalg
