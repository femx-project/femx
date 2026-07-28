#pragma once

#include <mpi.h>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/native/HostVectorHandler.hpp>

namespace femx::linalg
{

/** @brief Own MPI execution resources for Host assembly. */
class MpiContext final : public Context<MemorySpace::Host>
{
public:
  /**
   * @brief Duplicate and own an MPI communicator.
   *
   * @param[in] comm - Communicator to duplicate.
   * @throws std::runtime_error - If MPI is unavailable or duplication fails.
   */
  explicit MpiContext(MPI_Comm comm = MPI_COMM_SELF);

  /** @brief Release the owned MPI communicator. */
  ~MpiContext() override;

  MpiContext(const MpiContext&)            = delete;
  MpiContext& operator=(const MpiContext&) = delete;
  MpiContext(MpiContext&&)                 = delete;
  MpiContext& operator=(MpiContext&&)      = delete;

  /** @brief Return the owned Host vector operations. */
  HostVectorHandler& vectorHandler() noexcept override;

  /**
   * @brief Return this rank's contiguous element range.
   *
   * @param[in] count - Global element count.
   * @return Rank-local half-open element range.
   * @throws std::runtime_error - If `count` is negative or MPI fails.
   */
  IndexRange elementRange(Index count) const override;

  /**
   * @brief Sum replicated Host values over the communicator.
   *
   * @param[in,out] vals - Values replaced by their communicator-wide sums.
   * @throws std::runtime_error - If MPI fails.
   */
  void allReduceSum(HostVectorView<Real> vals) const override;

  /** @brief Complete pending Host work; MPI operations are synchronous here. */
  void sync() const override;

  /** @brief Return the owned MPI communicator. */
  MPI_Comm comm() const noexcept;

private:
  MPI_Comm          comm_{MPI_COMM_NULL}; ///< Owned MPI communicator.
  HostVectorHandler vec_handler_;         ///< Owned Host vector operations.
};

} // namespace femx::linalg
