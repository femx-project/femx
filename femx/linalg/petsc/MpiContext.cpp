#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>
#include <femx/linalg/petsc/PETScPartition.hpp>

namespace femx::linalg
{
namespace
{

void checkMPI(int status, const char* operation)
{
  if (status != MPI_SUCCESS)
  {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

IndexRange contiguousRange(Index count, int rank, int size)
{
  const Index rank_index = static_cast<Index>(rank);
  const Index comm_size  = static_cast<Index>(size);
  const Index base       = count / comm_size;
  const Index extra      = count % comm_size;
  const Index begin =
      rank_index * base + std::min(rank_index, extra);
  const Index width = base + (rank_index < extra ? 1 : 0);
  return {begin, begin + width};
}

} // namespace

MpiContext::MpiContext(MPI_Comm comm)
{
  int initialized = 0;
  checkMPI(MPI_Initialized(&initialized), "MPI_Initialized");

  require(initialized != 0, "MpiContext requires initialized MPI");
  require(comm != MPI_COMM_NULL, "MpiContext requires a valid communicator");

  checkMPI(MPI_Comm_dup(comm, &comm_), "MPI_Comm_dup");
  checkMPI(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
  checkMPI(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");
}

MpiContext::~MpiContext()
{
  if (comm_ == MPI_COMM_NULL)
  {
    return;
  }

  int finalized = 0;
  if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0)
  {
    MPI_Comm_free(&comm_);
  }
}

VectorHandler<MemorySpace::Host>& MpiContext::vectorHandler() noexcept
{
  return vec_handler_;
}

MatrixHandler<MemorySpace::Host>& MpiContext::matrixHandler() noexcept
{
  return mat_handler_;
}

IndexRange MpiContext::elementRange(Index count) const
{
  require(count >= 0,
          "MpiContext element count must be nonnegative");

  if (partition_)
  {
    return {0, count};
  }

  return contiguousRange(count, rank_, size_);
}

bool MpiContext::ownsElement(
    Index                       element,
    Index                       count,
    HostVectorView<const Index> rows) const
{
  require(count >= 0 && element >= 0 && element < count,
          "MpiContext element index is out of range");

  if (partition_ && !rows.empty())
  {
    return partition_->elementOwner(rows)
           == static_cast<PetscMPIInt>(rank_);
  }

  const IndexRange range = contiguousRange(count, rank_, size_);
  return element >= range.begin && element < range.end;
}

void MpiContext::allReduceSum(HostVectorView<Real> vals) const
{
  if (vals.empty())
  {
    return;
  }
  checkMPI(MPI_Allreduce(MPI_IN_PLACE,
                         vals.data(),
                         static_cast<int>(vals.size()),
                         MPI_DOUBLE,
                         MPI_SUM,
                         comm_),
           "MPI_Allreduce");
}

void MpiContext::sync() const
{
}

MPI_Comm MpiContext::comm() const noexcept
{
  return comm_;
}

void MpiContext::setPartition(
    std::shared_ptr<const PETScPartition> partition) noexcept
{
  partition_ = std::move(partition);
}

} // namespace femx::linalg
