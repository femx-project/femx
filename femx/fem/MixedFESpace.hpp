#pragma once

#include <vector>

#include <femx/core/Types.hpp>
#include <femx/fem/FESpace.hpp>

namespace femx
{

class MixedFieldView
{
public:
  /** @brief Create a view of one field inside a mixed finite elem space. */
  MixedFieldView(const FESpace* space,
                 index_type     local_offset,
                 index_type     global_offset);

  // Accessor
  const FESpace& space() const noexcept;
  index_type     numComponents() const noexcept;
  index_type     numShapesPerElem() const noexcept;
  index_type     numDofsPerElem() const noexcept;

  /** @brief Return the mixed-space local dof index for a shape function component. */
  index_type localDof(index_type shape_index,
                      index_type component = 0) const noexcept;

  /** @brief Return the mixed-space global dof index for a scalar dof component. */
  index_type globalDof(index_type scalar_dof,
                       index_type component = 0) const noexcept;

private:
  const FESpace* space_{nullptr};
  index_type     local_offset_{0};
  index_type     global_offset_{0};
};

class MixedFESpace
{
public:
  /** @brief Add a finite elem field to the mixed space. */
  void addField(const FESpace& space);

  /** @brief Build offsets and dof maps for all fields in the mixed space. */
  void setup();

  // Accessor
  MixedFieldView field(index_type field_id) const;
  const Mesh&    mesh() const noexcept;
  index_type     numFields() const noexcept;
  index_type     numElems() const noexcept;
  index_type     numDofs() const noexcept;
  index_type     numDofsPerElem() const noexcept;

  /** @brief Fill the mixed-space global dof indices used by one elem. */
  void elemDofs(index_type               ic,
                std::vector<index_type>& dofs) const;

  /** @brief Return the mixed-space global dof indices used by one elem. */
  std::vector<index_type> elemDofs(index_type ic) const;

private:
  std::vector<FESpace>    fields_;
  std::vector<index_type> local_offsets_;
  std::vector<index_type> global_offsets_;
  index_type              num_dofs_per_elem_{0};
  index_type              num_dofs_{0};
};

} // namespace femx
