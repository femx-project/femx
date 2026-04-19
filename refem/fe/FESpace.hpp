#pragma once

#include <refem/common/Types.hpp>
#include <refem/fe/DofMap.hpp>
#include <refem/fe/FiniteElement.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

class FESpace
{
public:
  FESpace(const Mesh* mesh, const FiniteElement* finite_element);

  void setup();

  const Mesh&          mesh() const noexcept;
  const FiniteElement& finiteElement() const noexcept;
  const DofMap&        dofMap() const noexcept;
  index_type           numCells() const noexcept;
  index_type           numDofs() const noexcept;
  std::vector<index_type> cellDofs(index_type cell) const;

private:
  const Mesh*          mesh_{nullptr};
  const FiniteElement* finite_element_{nullptr};
  DofMap               dof_map_;
  index_type           num_dofs_{0};
};

} // namespace refem
