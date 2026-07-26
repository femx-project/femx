#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>

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
  DirichletControl();

  explicit DirichletControl(HostVector<Index> dofs);

  DirichletControl(HostVector<Index>                    state_dofs,
                   Index                                num_ctr_params,
                   HostVector<DirichletControlMapEntry> map_entries);

  Index numStateDofs() const;
  Index numControlParams() const;

  Index stateDof(Index i) const;

  const HostVector<Index>& stateDofs() const;
  const HostCsrMatrix&     matrix() const noexcept;

  /** Remove state dofs while preserving and compacting the map P. */
  DirichletControl withoutStateDofs(
      const HostVector<Index>& excluded) const;

  /** Compute P * direction in local controlled-state ordering. */
  void apply(const HostVector<Real>& dir, HostVector<Real>& out) const;

  /** Compute P^T * direction. */
  void applyTranspose(const HostVector<Real>& dir, HostVector<Real>& out) const;

private:
  void checkDofIndex(Index i) const;
  void checkControlVector(const HostVector<Real>& ctr) const;
  void checkStateVector(const HostVector<Real>& state) const;

private:
  HostVector<Index> dofs_;
  HostCsrMatrix     matrix_;
};

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    Index               ptag);

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname);

/** One scalar normal-velocity parameter per node on a boundary. */
DirichletControl makeNormalVelocityControl(
    const MixedFESpace&     space,
    Index                   ptag,
    const HostVector<Real>& nrm);

/** One scalar normal-velocity parameter per node on a boundary. */
DirichletControl makeNormalVelocityControl(
    const MixedFESpace&     space,
    const std::string&      pname,
    const HostVector<Real>& nrm);

} // namespace fem
} // namespace femx
