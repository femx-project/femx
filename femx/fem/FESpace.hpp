#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/fem/DofMap.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{

class FESpace
{
public:
  /** @brief Create a finite elem space on a mesh with the given component count. */
  FESpace(const Mesh*          mesh,
          const FiniteElement* finite_element,
          index_type           components = 1);

  /** @brief Build the elem-to-global-dof map for the space. */
  void setup();

  // Accessors 
  const Mesh&          mesh() const noexcept;
  const FiniteElement& finiteElement() const noexcept;
  const DofMap&        dofMap() const noexcept;
  index_type           numElems() const noexcept;
  index_type           numDofs() const noexcept;
  index_type           numComponents() const noexcept;
  index_type           numShapesPerElem() const noexcept;
  index_type           numDofsPerElem() const noexcept;

  /** @brief Return the local dof index for a shape function component. */
  index_type localDof(index_type shape_index,
                      index_type component) const noexcept;

  /** @brief Return the global dof index for a mesh node component. */
  index_type globalDof(index_type node,
                       index_type component) const noexcept;

  /** @brief Fill the global dof indices used by one elem. */
  void elemDofs(index_type               ic,
                std::vector<index_type>& dofs) const;

  /** @brief Return the global dof indices used by one elem. */
  std::vector<index_type> elemDofs(index_type ic) const;

private:
  const Mesh*          mesh_{nullptr};
  const FiniteElement* finite_element_{nullptr};
  DofMap               dof_map_;
  index_type           components_{1};
  index_type           num_shapes_per_elem_{0};
  index_type           num_dofs_{0};
};

} // namespace femx
