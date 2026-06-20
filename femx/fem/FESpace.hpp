#pragma once

#include <femx/core/Types.hpp>
#include <femx/fem/DofMap.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/fem/Mesh.hpp>

namespace femx
{

class FESpace
{
public:
  /** @brief Create a finite elem space on a mesh with the given component count. */
  FESpace(const Mesh*          mesh,
          const FiniteElement* finite_element,
          Index                components = 1);

  /** @brief Build the elem-to-global-dof map for the space. */
  void setup();

  // Accessors
  const Mesh&          mesh() const noexcept;
  const FiniteElement& finiteElement() const noexcept;
  const DofMap&        dofMap() const noexcept;
  Index                numElems() const noexcept;
  Index                numDofs() const noexcept;
  Index                numComponents() const noexcept;
  Index                numShapesPerElem() const noexcept;
  Index                numDofsPerElem() const noexcept;

  /** @brief Return the local dof index for a shape function component. */
  Index localDof(Index shape_index,
                 Index component) const noexcept;

  /** @brief Return the global dof index for a mesh node component. */
  Index globalDof(Index in,
                  Index component) const noexcept;

  /** @brief Fill the global dof indices used by one elem. */
  void elemDofs(Index          ic,
                Vector<Index>& dofs) const;

  /** @brief Return the global dof indices used by one elem. */
  Vector<Index> elemDofs(Index ic) const;

private:
  const Mesh*          mesh_{nullptr};
  const FiniteElement* fe_{nullptr};
  DofMap               dof_map_;
  Index                components_{1};
  Index                num_shapes_per_elem_{0};
  Index                num_dofs_{0};
};

} // namespace femx
