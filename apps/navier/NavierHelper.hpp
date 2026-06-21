#pragma once

#include <memory>

#include "NavierConfig.hpp"
#include <femx/linalg/Vector.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>

namespace femx::navier
{

std::unique_ptr<FiniteElement> makeElement(const Mesh& mesh);

MixedFESpace makeSpace(Mesh& mesh, FiniteElement& elem);

Point3 selectorCenter(const Mesh& mesh, const BoundarySelector& selector);

Vector<Index> gaugeDofs(const MixedFESpace&     space,
                        const BoundarySelector& selector);

DirichletControl makeVelocityControl(
    const MixedFESpace&     space,
    const BoundarySelector& selector);

} // namespace femx::navier
