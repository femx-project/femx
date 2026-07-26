#include <algorithm>
#include <stdexcept>
#include <string>

#include <femx/common/Checks.hpp>
#include <femx/linalg/petsc/MpiContext.hpp>

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

} // namespace

MpiContext::MpiContext(MPI_Comm comm)
{
  int initialized = 0;
  checkMPI(MPI_Initialized(&initialized), "MPI_Initialized");
  require(initialized != 0,
          "MpiContext requires initialized MPI");
  require(comm != MPI_COMM_NULL,
          "MpiContext requires a valid communicator");
  checkMPI(MPI_Comm_dup(comm, &comm_), "MPI_Comm_dup");
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

HostVectorHandler& MpiContext::vectors() noexcept
{
  return vectors_;
}

IndexRange MpiContext::elementRange(Index count) const
{
  require(count >= 0,
          "MpiContext element count must be nonnegative");

  int rank = 0;
  int size = 1;
  checkMPI(MPI_Comm_rank(comm_, &rank), "MPI_Comm_rank");
  checkMPI(MPI_Comm_size(comm_, &size), "MPI_Comm_size");

  const Index rank_index = static_cast<Index>(rank);
  const Index comm_size  = static_cast<Index>(size);
  const Index base       = count / comm_size;
  const Index extra      = count % comm_size;
  const Index begin =
      rank_index * base + std::min(rank_index, extra);
  const Index width = base + (rank_index < extra ? 1 : 0);
  return {begin, begin + width};
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

} // namespace femx::linalg
