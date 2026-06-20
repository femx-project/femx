#include <set>
#include <stdexcept>
#include <string>

#include <femx/fem/BoundaryCondition.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/algebra/SparseMatrix.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{

void DirichletCondition::addDof(Index dof, Real value)
{
  dofs_.push_back(dof);
  values_.push_back(value);
}

void DirichletCondition::addBoundary(const FESpace& space,
                                     Index          physical_tag,
                                     Real           value,
                                     Real           time,
                                     Index          component)
{
  addBoundary(space, physical_tag, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const FESpace&       space,
                                     Index                physical_tag,
                                     const BoundaryValue& value,
                                     Real                 time,
                                     Index                component)
{
  if (component < 0 || component >= space.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh&     mesh = space.mesh();
  std::set<Index> nodes;
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

  for (Index in : nodes)
  {
    const auto& point = mesh.node(in);
    addDof(space.globalDof(in, component), value(point, time));
  }
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     Index                 physical_tag,
                                     Real                  value,
                                     Real                  time,
                                     Index                 component)
{
  addBoundary(field, physical_tag, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     Index                 physical_tag,
                                     const BoundaryValue&  value,
                                     Real                  time,
                                     Index                 component)
{
  if (component < 0 || component >= field.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh&     mesh = field.space().mesh();
  std::set<Index> nodes;
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

  for (Index in : nodes)
  {
    const auto& point = mesh.node(in);
    addDof(field.globalDof(in, component), value(point, time));
  }
}

void DirichletCondition::addBoundary(const FESpace&        space,
                                     const BoundaryMarker& marker,
                                     Real                  value,
                                     Real                  time,
                                     Index                 component)
{
  addBoundary(space, marker, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const FESpace&        space,
                                     const BoundaryMarker& marker,
                                     const BoundaryValue&  value,
                                     Real                  time,
                                     Index                 component)
{
  if (component < 0 || component >= space.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh& mesh = space.mesh();
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& point = mesh.node(in);
    if (marker(point, time))
    {
      addDof(space.globalDof(in, component), value(point, time));
    }
  }
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     const BoundaryMarker& marker,
                                     Real                  value,
                                     Real                  time,
                                     Index                 component)
{
  addBoundary(field, marker, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              component);
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     const BoundaryMarker& marker,
                                     const BoundaryValue&  value,
                                     Real                  time,
                                     Index                 component)
{
  if (component < 0 || component >= field.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh& mesh = field.space().mesh();
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& point = mesh.node(in);
    if (marker(point, time))
    {
      addDof(field.globalDof(in, component), value(point, time));
    }
  }
}

const Vector<Index>& DirichletCondition::dofs() const noexcept
{
  return dofs_;
}

const Vector<Real>& DirichletCondition::values() const noexcept
{
  return values_;
}

void DirichletCondition::apply(SparseMatrix& A, Vector<Real>& b) const
{
  if (dofs_.size() != values_.size())
  {
    throw std::runtime_error("DirichletCondition has inconsistent data");
  }

  const Index* row_ptr = A.rowPtrData();
  const Index* col_ind = A.colIndData();
  Real*        values  = A.valuesData();

  std::vector<char> is_dirichlet(static_cast<std::size_t>(A.rows()), 0);
  std::vector<char> found_diagonal(static_cast<std::size_t>(A.rows()), 0);
  Vector<Real>      dirichlet_values(A.rows());

  for (Index c = 0; c < dofs_.size(); ++c)
  {
    const Index dof   = dofs_[c];
    const Real  value = values_[c];

    if (dof < 0 || dof >= A.rows() || dof >= b.size())
    {
      throw std::runtime_error("Dirichlet dof is out of range");
    }

    is_dirichlet[static_cast<std::size_t>(dof)] = 1;
    dirichlet_values[dof]                       = value;
  }

  for (Index row = 0; row < A.rows(); ++row)
  {
    const bool row_is_dirichlet =
        is_dirichlet[static_cast<std::size_t>(row)] != 0;

    for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      const Index col = col_ind[k];

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
        b[row]    -= values[k] * dirichlet_values[col];
        values[k]  = 0.0;
      }
    }

    if (row_is_dirichlet)
    {
      b[row] = dirichlet_values[row];
    }
  }

  for (Index dof = 0; dof < A.rows(); ++dof)
  {
    if (is_dirichlet[static_cast<std::size_t>(dof)] != 0 && found_diagonal[static_cast<std::size_t>(dof)] == 0)
    {
      throw std::runtime_error("Dirichlet row has no diagonal entry");
    }
  }
}

} // namespace femx
