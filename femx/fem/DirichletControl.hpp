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

  explicit DirichletControl(Vector<Index> dofs);

  DirichletControl(Vector<Index>                    state_dofs,
                   Index                            num_ctr_params,
                   Vector<DirichletControlMapEntry> map_entries);

  Index numStateDofs() const;
  Index numControlParams() const;

  Index stateDof(Index i) const;

  const Vector<Index>&                    stateDofs() const;
  const Vector<DirichletControlMapEntry>& mapEntries() const;

  /** Remove state dofs while preserving and compacting the map P. */
  DirichletControl withoutStateDofs(
      const Vector<Index>& excluded) const;

  /** Compute P * direction in local controlled-state ordering. */
  void apply(const Vector<Real>& direction, Vector<Real>& out) const;

  /** Compute P^T * direction. */
  void applyTranspose(const Vector<Real>& direction, Vector<Real>& out) const;

private:
  void checkDofIndex(Index i) const;
  void checkControlVector(const Vector<Real>& control) const;
  void checkStateVector(const Vector<Real>& state) const;

private:
  Vector<Index>                    dofs_;
  Index                            num_ctr_params_{0};
  Vector<DirichletControlMapEntry> map_entries_;
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
    const Vector<Real>& normal);

/** One scalar normal-velocity parameter per node on a boundary. */
DirichletControl makeNormalVelocityControl(
    const MixedFESpace& space,
    const std::string&  pname,
    const Vector<Real>& normal);

} // namespace fem
} // namespace femx
