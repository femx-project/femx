#pragma once

#include <mpi.h>
#include <petscsystypes.h>

#include <memory>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

/**
 * @brief Map application degrees of freedom to a graph-partitioned PETSc layout.
 */
class PETScPartition final
{
public:
  /**
   * @brief Partition a square Host CSR graph over a communicator.
   *
   * ParMETIS is preferred when PETSc provides it. The PETSc options database
   * can override the partitioner with `-femx_mat_partitioning_type`.
   *
   * @param[in] comm - Communicator receiving the partitioned graph.
   * @param[in] pattern - Replicated square application-order CSR graph.
   * @return Shared immutable partition and numbering.
   * @throws std::runtime_error - If the graph is invalid or PETSc or MPI
   * reports an error.
   */
  static std::shared_ptr<const PETScPartition> create(
      MPI_Comm              comm,
      const HostCsrPattern& pattern);

  /** @brief Return the number of global degrees of freedom. */
  Index size() const noexcept;

  /** @brief Return the number of degrees of freedom owned by this rank. */
  PetscInt localSize() const noexcept;

  /** @brief Return the first PETSc index owned by this rank. */
  PetscInt begin() const noexcept;

  /** @brief Return one past the last PETSc index owned by this rank. */
  PetscInt end() const noexcept;

  /**
   * @brief Map an application index to the partitioned PETSc numbering.
   *
   * @param[in] index - Application degree-of-freedom index.
   * @return Partitioned PETSc index.
   * @throws std::runtime_error - If `index` is out of range.
   */
  PetscInt petscIndex(Index index) const;

  /**
   * @brief Map a partitioned PETSc index to the application numbering.
   *
   * @param[in] index - Partitioned PETSc index.
   * @return Application degree-of-freedom index.
   * @throws std::runtime_error - If `index` is out of range.
   */
  Index applicationIndex(PetscInt index) const;

  /**
   * @brief Return the rank owning an application degree of freedom.
   *
   * @param[in] index - Application degree-of-freedom index.
   * @return Owning communicator rank.
   * @throws std::runtime_error - If `index` is out of range.
   */
  PetscMPIInt owner(Index index) const;

  /**
   * @brief Select the rank owning the largest share of an element's rows.
   *
   * Ties are assigned to the lower rank.
   *
   * @param[in] rows - Element rows in application numbering.
   * @return Rank assigned to the element.
   * @throws std::runtime_error - If `rows` is empty or contains an invalid
   * index.
   */
  PetscMPIInt elementOwner(HostVectorView<const Index> rows) const;

  /** @brief Return the PETSc partitioner type used to build this layout. */
  const std::string& type() const noexcept;

private:
  PETScPartition() = default;

  Index                   size_{0};       ///< Global degree-of-freedom count.
  PetscInt                local_size_{0}; ///< Rank-local degree-of-freedom count.
  PetscInt                begin_{0};      ///< First rank-local PETSc index.
  PetscMPIInt             comm_size_{1};  ///< Number of communicator ranks.
  HostVector<PetscInt>    app_to_petsc_;  ///< Application-to-PETSc permutation.
  HostVector<Index>       petsc_to_app_;  ///< PETSc-to-application permutation.
  HostVector<PetscMPIInt> owners_;        ///< Owner rank by application index.
  std::string             type_;          ///< PETSc partitioner type.
};

} // namespace femx::linalg
