#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

class MixedFESpace;

/** One nonzero entry of the local control map P. */
struct DirichletControlMapEntry
{
  Index state_row = 0; ///< Row in the controlled state-dof vector.
  Index ctr_col   = 0; ///< Column in the control-parameter vector.
  Real  weight    = 0.0;
};

/** @brief Linear map u_D = P q for selected Dirichlet state dofs. */
class DirichletControl
{
public:
  DirichletControl() = default;

  explicit DirichletControl(Array<Index> dofs);

  DirichletControl(Array<Index>                    state_dofs,
                   Index                           num_ctr_params,
                   Array<DirichletControlMapEntry> map_entries);

  Index numStateDofs() const;
  Index numControlParams() const;

  Index stateDof(Index i) const;

  const Array<Index>&                    stateDofs() const;
  const Array<DirichletControlMapEntry>& mapEntries() const;

  /** Remove state dofs while preserving and compacting the map P. */
  DirichletControl withoutStateDofs(
      const Array<Index>& excluded) const;

  /** Compute P * direction in local controlled-state ordering. */
  void apply(const HostVector& direction, HostVector& out) const;

  /** Compute P^T * direction. */
  void applyTranspose(const HostVector& direction, HostVector& out) const;

private:
  void checkDofIndex(Index i) const;
  void checkControlVector(const HostVector& control) const;
  void checkStateVector(const HostVector& state) const;

private:
  Array<Index>                    dofs_;
  Index                           num_ctr_params_{0};
  Array<DirichletControlMapEntry> map_entries_;
};

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    Index               ptag);

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname);

/** One scalar normal-velocity parameter per node on a boundary. */
DirichletControl makeNormalVelocityControl(
    const MixedFESpace& space,
    Index               ptag,
    const HostVector&   normal);

/** One scalar normal-velocity parameter per node on a boundary. */
DirichletControl makeNormalVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname,
    const HostVector&   normal);

} // namespace fem
} // namespace femx
