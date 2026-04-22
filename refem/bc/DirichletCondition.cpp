#include <set>
#include <stdexcept>
#include <string>

#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/BlockFESpace.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

void DirichletCondition::addDof(index_type dof, real_type value)
{
  dofs_.push_back(dof);
  values_.push_back(value);
}

void DirichletCondition::addBoundary(const FESpace& space,
                                     index_type     physical_tag,
                                     real_type      value,
                                     real_type      time,
                                     index_type     component)
{
  addBoundary(space, physical_tag, [value](const Mesh::Node&, real_type)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const FESpace&       space,
                                     index_type           physical_tag,
                                     const BoundaryValue& value,
                                     real_type            time,
                                     index_type           component)
{
  if (component < 0 || component >= space.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh&          mesh = space.mesh();
  std::set<index_type> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.physical_tag == physical_tag)
    {
      nodes.insert(facet.node_ids.begin(), facet.node_ids.end());
    }
  }

  if (nodes.empty())
  {
    throw std::runtime_error(
        "No boundary facets found for physical tag " + std::to_string(physical_tag));
  }

  for (index_type node : nodes)
  {
    const auto& point = mesh.node(node);
    addDof(space.globalDof(node, component), value(point, time));
  }
}

void DirichletCondition::addBoundary(const BlockFieldView& field,
                                     index_type            physical_tag,
                                     real_type             value,
                                     real_type             time,
                                     index_type            component)
{
  addBoundary(field, physical_tag, [value](const Mesh::Node&, real_type)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const BlockFieldView& field,
                                     index_type            physical_tag,
                                     const BoundaryValue&  value,
                                     real_type             time,
                                     index_type            component)
{
  if (component < 0 || component >= field.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh&          mesh = field.space().mesh();
  std::set<index_type> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.physical_tag == physical_tag)
    {
      nodes.insert(facet.node_ids.begin(), facet.node_ids.end());
    }
  }

  if (nodes.empty())
  {
    throw std::runtime_error(
        "No boundary facets found for physical tag " + std::to_string(physical_tag));
  }

  for (index_type node : nodes)
  {
    const auto& point = mesh.node(node);
    addDof(field.globalDof(node, component), value(point, time));
  }
}

void DirichletCondition::addBoundary(const FESpace&        space,
                                     const BoundaryMarker& marker,
                                     real_type             value,
                                     real_type             time,
                                     index_type            component)
{
  addBoundary(space, marker, [value](const Mesh::Node&, real_type)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const FESpace&        space,
                                     const BoundaryMarker& marker,
                                     const BoundaryValue&  value,
                                     real_type             time,
                                     index_type            component)
{
  if (component < 0 || component >= space.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh& mesh = space.mesh();
  for (index_type in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& point = mesh.node(in);
    if (marker(point, time))
    {
      addDof(space.globalDof(in, component), value(point, time));
    }
  }
}

void DirichletCondition::addBoundary(const BlockFieldView& field,
                                     const BoundaryMarker& marker,
                                     real_type             value,
                                     real_type             time,
                                     index_type            component)
{
  addBoundary(field, marker, [value](const Mesh::Node&, real_type)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const BlockFieldView& field,
                                     const BoundaryMarker& marker,
                                     const BoundaryValue&  value,
                                     real_type             time,
                                     index_type            component)
{
  if (component < 0 || component >= field.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh& mesh = field.space().mesh();
  for (index_type in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& point = mesh.node(in);
    if (marker(point, time))
    {
      addDof(field.globalDof(in, component), value(point, time));
    }
  }
}

const std::vector<index_type>& DirichletCondition::dofs() const noexcept
{
  return dofs_;
}

const std::vector<real_type>& DirichletCondition::values() const noexcept
{
  return values_;
}

void DirichletCondition::apply(SparseMatrix& A, Vector& b) const
{
  if (dofs_.size() != values_.size())
  {
    throw std::runtime_error("DirichletCondition has inconsistent data");
  }

  const index_type* row_ptr = A.rowPtrData();
  const index_type* col_ind = A.colIndData();
  real_type*        values  = A.valuesData();

  std::vector<char>      is_dirichlet(static_cast<std::size_t>(A.rows()), 0);
  std::vector<char>      found_diagonal(static_cast<std::size_t>(A.rows()), 0);
  std::vector<real_type> dirichlet_values(static_cast<std::size_t>(A.rows()), 0.0);

  for (std::size_t c = 0; c < dofs_.size(); ++c)
  {
    const index_type dof   = dofs_[c];
    const real_type  value = values_[c];

    if (dof < 0 || dof >= A.rows() || dof >= b.size())
    {
      throw std::runtime_error("Dirichlet dof is out of range");
    }

    is_dirichlet[static_cast<std::size_t>(dof)]     = 1;
    dirichlet_values[static_cast<std::size_t>(dof)] = value;
  }

  for (index_type row = 0; row < A.rows(); ++row)
  {
    const bool row_is_dirichlet =
        is_dirichlet[static_cast<std::size_t>(row)] != 0;

    for (index_type k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      const index_type col = col_ind[k];

      if (row_is_dirichlet)
      {
        values[k] = 0.0;
        if (col == row)
        {
          values[k]                                     = 1.0;
          found_diagonal[static_cast<std::size_t>(row)] = 1;
        }
      }
      else if (is_dirichlet[static_cast<std::size_t>(col)] != 0)
      {
        b[row]    -= values[k] * dirichlet_values[static_cast<std::size_t>(col)];
        values[k]  = 0.0;
      }
    }

    if (row_is_dirichlet)
    {
      b[row] = dirichlet_values[static_cast<std::size_t>(row)];
    }
  }

  for (index_type dof = 0; dof < A.rows(); ++dof)
  {
    if (is_dirichlet[static_cast<std::size_t>(dof)] != 0 && found_diagonal[static_cast<std::size_t>(dof)] == 0)
    {
      throw std::runtime_error("Dirichlet row has no diagonal entry");
    }
  }
}

} // namespace refem
