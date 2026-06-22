#pragma once

#include <memory>

#include "NavierConfig.hpp"
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::navier
{

std::unique_ptr<FiniteElement> makeElement(const Mesh& mesh);

MixedFESpace makeSpace(Mesh& mesh, FiniteElement& elem);

Point3 selectorCenter(const Mesh& mesh, const BoundarySelector& sel);

Vector<Index> gaugeDofs(const MixedFESpace&     space,
                        const BoundarySelector& sel);

DirichletControl makeVelocityControl(
    const MixedFESpace&     space,
    const BoundarySelector& sel);

} // namespace femx::navier
