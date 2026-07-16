#pragma once

#include <memory>

#include "Config.hpp"
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx::model::ns
{

std::unique_ptr<fem::FiniteElement> makeElement(const fem::Mesh& mesh);

fem::MixedFESpace makeSpace(fem::Mesh& mesh, fem::FiniteElement& elem);

Point3 selectorCenter(const fem::Mesh& mesh, const BoundarySelector& sel);

Vector<Index> gaugeDofs(const fem::MixedFESpace& space,
                        const BoundarySelector&  sel);

fem::DirichletControl makeVelocityControl(
    const fem::MixedFESpace& space,
    const BoundarySelector&  sel);

void splitStateFields(VectorView<const Real>   state,
                      const fem::MixedFESpace& space,
                      Vector<Real>&            ux,
                      Vector<Real>&            uy,
                      Vector<Real>&            uz,
                      Vector<Real>&            pressure);

} // namespace femx::model::ns
