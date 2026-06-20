#pragma once

#include <string>

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{

class MixedFESpace;

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
