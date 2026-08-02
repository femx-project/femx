#include <petscis.h>
#include <petscmat.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <femx/common/Checks.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/petsc/PETScPartition.hpp>
#include <femx/linalg/petsc/PETScUtilities.hpp>

namespace femx::linalg
{
namespace
{

using detail::check;
using detail::checkMPI;

class ScopedMat
{
public:
  ~ScopedMat()
  {
    if (mat_ != nullptr)
    {
      MatDestroy(&mat_);
    }
  }

  Mat get() const
  {
    return mat_;
  }

  Mat* put()
  {
    return &mat_;
  }

private:
  Mat mat_{nullptr};
};

class ScopedPartitioning
{
public:
  ~ScopedPartitioning()
  {
    if (partitioning_ != nullptr)
    {
      MatPartitioningDestroy(&partitioning_);
    }
  }

  MatPartitioning get() const
  {
    return partitioning_;
  }

  MatPartitioning* put()
  {
    return &partitioning_;
  }

private:
  MatPartitioning partitioning_{nullptr};
};

class ScopedIS
{
public:
  ~ScopedIS()
  {
    if (is_ != nullptr)
    {
      ISDestroy(&is_);
    }
  }

  IS get() const
  {
    return is_;
  }

  IS* put()
  {
    return &is_;
  }

private:
  IS is_{nullptr};
};

void initialOwnership(MPI_Comm  comm,
                      Index     size,
                      PetscInt& local_size,
                      PetscInt& begin)
{
  local_size           = PETSC_DECIDE;
  PetscInt global_size = static_cast<PetscInt>(size);
  check(PetscSplitOwnership(comm, &local_size, &global_size),
        "PetscSplitOwnership");

  begin = 0;
  checkMPI(MPI_Exscan(&local_size,
                      &begin,
                      1,
                      MPIU_INT,
                      MPI_SUM,
                      comm),
           "MPI_Exscan");
  PetscMPIInt rank = 0;
  checkMPI(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
  if (rank == 0)
  {
    begin = 0;
  }
}

void makeAdjacency(MPI_Comm              comm,
                   const HostCsrPattern& pattern,
                   ScopedMat&            adjacency)
{
  PetscInt local_size = 0;
  PetscInt begin      = 0;
  initialOwnership(comm, pattern.rows(), local_size, begin);
  const PetscInt end = begin + local_size;

  const Index* row_ptr = pattern.rowPtrData();
  const Index* col_ind = pattern.colIndData();

  std::vector<std::vector<PetscInt>> neighbors(static_cast<std::size_t>(local_size));
  for (Index i = 0; i < pattern.rows(); ++i)
  {
    for (Index k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
    {
      const Index column = col_ind[k];
      if (column == i)
      {
        continue;
      }
      if (i >= begin && i < end)
      {
        neighbors[static_cast<std::size_t>(i - begin)].push_back(
            static_cast<PetscInt>(column));
      }
      if (column >= begin && column < end)
      {
        neighbors[static_cast<std::size_t>(column - begin)].push_back(
            static_cast<PetscInt>(i));
      }
    }
  }

  PetscInt num_edges = 0;
  for (auto& row_neighbors : neighbors)
  {
    std::sort(row_neighbors.begin(), row_neighbors.end());
    row_neighbors.erase(
        std::unique(row_neighbors.begin(), row_neighbors.end()),
        row_neighbors.end());
    num_edges += static_cast<PetscInt>(row_neighbors.size());
  }

  PetscInt* offsets = nullptr;
  PetscInt* columns = nullptr;
  check(PetscMalloc1(static_cast<std::size_t>(local_size) + 1, &offsets), "PetscMalloc1");
  const PetscErrorCode alloc_error = PetscMalloc1(static_cast<std::size_t>(num_edges), &columns);
  if (alloc_error != PETSC_SUCCESS)
  {
    PetscFree(offsets);
    check(alloc_error, "PetscMalloc1");
  }

  PetscInt edge = 0;
  offsets[0]    = 0;
  for (PetscInt i = 0; i < local_size; ++i)
  {
    for (const PetscInt j :
         neighbors[static_cast<std::size_t>(i)])
    {
      columns[edge++] = j;
    }
    offsets[i + 1] = edge;
  }

  const PetscErrorCode ierr =
      MatCreateMPIAdj(comm,
                      local_size,
                      static_cast<PetscInt>(pattern.cols()),
                      offsets,
                      columns,
                      nullptr,
                      adjacency.put());
  if (ierr != PETSC_SUCCESS)
  {
    PetscFree(offsets);
    PetscFree(columns);
    check(ierr, "MatCreateMPIAdj");
  }
}

void setDefaultType(MatPartitioning partitioning)
{
#if defined(PETSC_HAVE_PARMETIS)
  check(MatPartitioningSetType(partitioning,
                               MATPARTITIONINGPARMETIS),
        "MatPartitioningSetType");
#elif defined(PETSC_HAVE_PTSCOTCH)
  check(MatPartitioningSetType(partitioning,
                               MATPARTITIONINGPTSCOTCH),
        "MatPartitioningSetType");
#else
  check(MatPartitioningSetType(partitioning,
                               MATPARTITIONINGAVERAGE),
        "MatPartitioningSetType");
#endif
}

void gatherIS(IS local, ScopedIS& global)
{
  check(ISAllGather(local, global.put()), "ISAllGather");
}

} // namespace

std::shared_ptr<const PETScPartition> PETScPartition::create(
    MPI_Comm              comm,
    const HostCsrPattern& pattern)
{
  require(pattern.rows() == pattern.cols(),
          "PETSc graph partitioning requires a square CSR pattern");

  auto out   = std::shared_ptr<PETScPartition>(new PETScPartition);
  out->size_ = pattern.rows();

  PetscMPIInt rank = 0;
  checkMPI(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
  checkMPI(MPI_Comm_size(comm, &out->comm_size_), "MPI_Comm_size");

  out->app_to_petsc_.resize(out->size_);
  out->petsc_to_app_.resize(out->size_);
  out->owners_.resize(out->size_);

  if (out->size_ == 0 || out->comm_size_ == 1)
  {
    out->local_size_ = out->comm_size_ == 1 ? static_cast<PetscInt>(out->size_) : 0;
    out->begin_      = 0;
    out->type_       = out->size_ == 0 ? "empty" : "single";
    for (Index i = 0; i < out->size_; ++i)
    {
      out->app_to_petsc_[i] = static_cast<PetscInt>(i);
      out->petsc_to_app_[i] = i;
      out->owners_[i]       = 0;
    }
    return out;
  }

  ScopedMat          adjacency;
  ScopedPartitioning partitioning;
  ScopedIS           local_owners;
  ScopedIS           local_numbering;
  ScopedIS           global_owners;
  ScopedIS           global_numbering;

  makeAdjacency(comm, pattern, adjacency);
  check(MatPartitioningCreate(comm, partitioning.put()),
        "MatPartitioningCreate");
  check(MatPartitioningSetAdjacency(partitioning.get(),
                                    adjacency.get()),
        "MatPartitioningSetAdjacency");
  check(MatPartitioningSetNParts(partitioning.get(), out->comm_size_),
        "MatPartitioningSetNParts");
  setDefaultType(partitioning.get());
  check(PetscObjectSetOptionsPrefix(
            reinterpret_cast<PetscObject>(partitioning.get()),
            "femx_"),
        "PetscObjectSetOptionsPrefix");
  check(MatPartitioningSetFromOptions(partitioning.get()),
        "MatPartitioningSetFromOptions");
  check(MatPartitioningApply(partitioning.get(),
                             local_owners.put()),
        "MatPartitioningApply");
  check(ISPartitioningToNumbering(local_owners.get(),
                                  local_numbering.put()),
        "ISPartitioningToNumbering");

  HostVector<PetscInt> counts(out->comm_size_);
  check(ISPartitioningCount(local_owners.get(),
                            out->comm_size_,
                            counts.data()),
        "ISPartitioningCount");

  out->local_size_ = counts[rank];
  out->begin_      = 0;
  for (PetscMPIInt irank = 0; irank < rank; ++irank)
  {
    out->begin_ += counts[irank];
  }

  gatherIS(local_owners.get(), global_owners);
  gatherIS(local_numbering.get(), global_numbering);

  PetscInt owners_size    = 0;
  PetscInt numbering_size = 0;
  check(ISGetLocalSize(global_owners.get(), &owners_size),
        "ISGetLocalSize");
  check(ISGetLocalSize(global_numbering.get(), &numbering_size),
        "ISGetLocalSize");
  require(owners_size == out->size_ && numbering_size == out->size_,
          "PETSc partitioning returned an incompatible global numbering");

  const PetscInt*           owners    = nullptr;
  const PetscInt*           numbering = nullptr;
  HostVector<unsigned char> seen(out->size_, 0);
  check(ISGetIndices(global_owners.get(), &owners),
        "ISGetIndices");
  check(ISGetIndices(global_numbering.get(), &numbering),
        "ISGetIndices");
  for (Index i = 0; i < out->size_; ++i)
  {
    const PetscInt new_index = numbering[i];
    require(new_index >= 0 && new_index < out->size_,
            "PETSc partitioning returned an invalid global index");
    require(owners[i] >= 0 && owners[i] < out->comm_size_,
            "PETSc partitioning returned an invalid owner rank");
    require(seen[new_index] == 0,
            "PETSc partitioning returned duplicate global indices");
    seen[new_index]               = 1;
    out->app_to_petsc_[i]         = new_index;
    out->petsc_to_app_[new_index] = i;
    out->owners_[i]               = static_cast<PetscMPIInt>(owners[i]);
  }
  check(ISRestoreIndices(global_numbering.get(), &numbering),
        "ISRestoreIndices");
  check(ISRestoreIndices(global_owners.get(), &owners),
        "ISRestoreIndices");

  MatPartitioningType type = nullptr;
  check(MatPartitioningGetType(partitioning.get(), &type),
        "MatPartitioningGetType");
  out->type_ = type == nullptr ? "unknown" : type;

  return out;
}

Index PETScPartition::size() const noexcept
{
  return size_;
}

PetscInt PETScPartition::localSize() const noexcept
{
  return local_size_;
}

PetscInt PETScPartition::begin() const noexcept
{
  return begin_;
}

PetscInt PETScPartition::end() const noexcept
{
  return begin_ + local_size_;
}

PetscInt PETScPartition::petscIndex(Index index) const
{
  require(index >= 0 && index < size_,
          "PETSc partition application index is out of range");
  return app_to_petsc_[index];
}

Index PETScPartition::applicationIndex(PetscInt index) const
{
  require(index >= 0 && index < size_,
          "PETSc partition global index is out of range");
  return petsc_to_app_[index];
}

PetscMPIInt PETScPartition::owner(Index index) const
{
  require(index >= 0 && index < size_,
          "PETSc partition owner index is out of range");
  return owners_[index];
}

PetscMPIInt PETScPartition::elementOwner(
    HostVectorView<const Index> rows) const
{
  require(!rows.empty(),
          "PETSc partition cannot assign an element without rows");

  PetscMPIInt best_owner = owner(rows[0]);
  Index       best_count = 0;
  for (Index i = 0; i < rows.size(); ++i)
  {
    const PetscMPIInt candidate = owner(rows[i]);
    Index             count     = 0;
    for (Index j = 0; j < rows.size(); ++j)
    {
      if (owner(rows[j]) == candidate)
      {
        ++count;
      }
    }
    if (count > best_count
        || (count == best_count && candidate < best_owner))
    {
      best_owner = candidate;
      best_count = count;
    }
  }
  return best_owner;
}

const std::string& PETScPartition::type() const noexcept
{
  return type_;
}

} // namespace femx::linalg
