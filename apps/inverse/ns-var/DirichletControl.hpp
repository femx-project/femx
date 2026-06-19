#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class DirichletControl
{
public:
  DirichletControl() = default;

  explicit DirichletControl(Vector<Index> dofs);

  Index numDofs() const;
  Index numParams(Index steps) const;
  Index stateDof(Index i) const;
  Index paramIndex(Index step, Index i) const;

  const Vector<Index>& stateDofs() const;

private:
  void checkDofIndex(Index i) const;

private:
  Vector<Index> dofs_;
};

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    Index               physical_tag);

DirichletControl makeVelocityControl(
    const MixedFESpace& space,
    const std::string&  physical_name);

} // namespace femx
