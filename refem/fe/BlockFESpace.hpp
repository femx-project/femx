#pragma once

#include <vector>

#include <refem/common/Types.hpp>
#include <refem/fe/FESpace.hpp>

namespace refem
{

class BlockFieldView
{
public:
  BlockFieldView(const FESpace* space,
                 index_type     local_offset,
                 index_type     global_offset);

  const FESpace& space() const noexcept;

  index_type numComponents() const noexcept;
  index_type numShapesPerElem() const noexcept;
  index_type numDofsPerElem() const noexcept;

  index_type localDof(index_type shape_index,
                      index_type component = 0) const noexcept;
  index_type globalDof(index_type scalar_dof,
                       index_type component = 0) const noexcept;

private:
  const FESpace* space_{nullptr};
  index_type     local_offset_{0};
  index_type     global_offset_{0};
};

class BlockFESpace
{
public:
  void addField(const FESpace& space);
  void setup();

  BlockFieldView field(index_type field_id) const;

  const Mesh& mesh() const noexcept;
  index_type  numFields() const noexcept;
  index_type  numElems() const noexcept;
  index_type  numDofs() const noexcept;
  index_type  numDofsPerElem() const noexcept;

  void elemDofs(index_type ic,
                std::vector<index_type>& dofs) const;
  std::vector<index_type> elemDofs(index_type ic) const;

private:
  std::vector<FESpace>    fields_;
  std::vector<index_type> local_offsets_;
  std::vector<index_type> global_offsets_;
  index_type              num_dofs_per_elem_{0};
  index_type              num_dofs_{0};
};

} // namespace refem
