#pragma once

#include <vector>

#include <refem/common/Types.hpp>
#include <refem/fe/DofMap.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

class FESpace
{
public:
  FESpace(const Mesh*          mesh,
          const FiniteElement* finite_element,
          index_type           components = 1);

  void setup();

  const Mesh&             mesh() const noexcept;
  const FiniteElement&    finiteElement() const noexcept;
  const DofMap&           dofMap() const noexcept;
  index_type              numElems() const noexcept;
  index_type              numDofs() const noexcept;
  index_type              numComponents() const noexcept;
  index_type              numShapesPerElem() const noexcept;
  index_type              numDofsPerElem() const noexcept;
  index_type              localDof(index_type shape_index,
                                   index_type component) const noexcept;
  index_type              globalDof(index_type node,
                                    index_type component) const noexcept;
  void                    elemDofs(index_type ic,
                                   std::vector<index_type>& dofs) const;
  std::vector<index_type> elemDofs(index_type ic) const;

private:
  const Mesh*          mesh_{nullptr};
  const FiniteElement* finite_element_{nullptr};
  DofMap               dof_map_;
  index_type           components_{1};
  index_type           num_shapes_per_elem_{0};
  index_type           num_dofs_{0};
};

} // namespace refem
