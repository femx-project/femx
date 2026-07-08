#include <set>
#include <stdexcept>
#include <string>

#include <femx/fem/BoundaryCondition.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

void DirichletCondition::addDof(Index id, Real value)
{
  dofs_.push_back(id);
  vals_.push_back(value);
}

void DirichletCondition::addBoundary(const FESpace& space,
                                     Index          ptag,
                                     Real           value,
                                     Real           time,
                                     Index          comp)
{
  addBoundary(space, ptag, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              comp);
}

void DirichletCondition::addBoundary(const FESpace&       space,
                                     Index                ptag,
                                     const BoundaryValue& value,
                                     Real                 time,
                                     Index                comp)
{
  if (comp < 0 || comp >= space.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh&     mesh = space.mesh();
  std::set<Index> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.ptag == ptag)
    {
      nodes.insert(facet.nids.begin(), facet.nids.end());
    }
  }

  if (nodes.empty())
  {
    throw std::runtime_error(
        "No boundary facets found for physical tag " + std::to_string(ptag));
  }

  for (Index in : nodes)
  {
    const auto& point = mesh.node(in);
    addDof(space.globalDof(in, comp), value(point, time));
  }
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     Index                 ptag,
                                     Real                  value,
                                     Real                  time,
                                     Index                 comp)
{
  addBoundary(field, ptag, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              comp);
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     Index                 ptag,
                                     const BoundaryValue&  value,
                                     Real                  time,
                                     Index                 comp)
{
  if (comp < 0 || comp >= field.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh&     mesh = field.space().mesh();
  std::set<Index> nodes;
  for (const auto& facet : mesh.boundaryFacets())
  {
    if (facet.ptag == ptag)
    {
      nodes.insert(facet.nids.begin(), facet.nids.end());
    }
  }

  if (nodes.empty())
  {
    throw std::runtime_error(
        "No boundary facets found for physical tag " + std::to_string(ptag));
  }

  for (Index in : nodes)
  {
    const auto& point = mesh.node(in);
    addDof(field.globalDof(in, comp), value(point, time));
  }
}

void DirichletCondition::addBoundary(const FESpace&        space,
                                     const BoundaryMarker& mark,
                                     Real                  value,
                                     Real                  time,
                                     Index                 comp)
{
  addBoundary(space, mark, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              comp);
}

void DirichletCondition::addBoundary(const FESpace&        space,
                                     const BoundaryMarker& mark,
                                     const BoundaryValue&  value,
                                     Real                  time,
                                     Index                 comp)
{
  if (comp < 0 || comp >= space.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh& mesh = space.mesh();
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& point = mesh.node(in);
    if (mark(point, time))
    {
      addDof(space.globalDof(in, comp), value(point, time));
    }
  }
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     const BoundaryMarker& mark,
                                     Real                  value,
                                     Real                  time,
                                     Index                 comp)
{
  addBoundary(field, mark, [value](const Mesh::Node&, Real)
              { return value; },
              time,
              comp);
}

void DirichletCondition::addBoundary(const MixedFieldView& field,
                                     const BoundaryMarker& mark,
                                     const BoundaryValue&  value,
                                     Real                  time,
                                     Index                 comp)
{
  if (comp < 0 || comp >= field.numComponents())
  {
    throw std::runtime_error("Dirichlet boundary component is out of range");
  }

  const Mesh& mesh = field.space().mesh();
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& point = mesh.node(in);
    if (mark(point, time))
    {
      addDof(field.globalDof(in, comp), value(point, time));
    }
  }
}

const Vector<Index>& DirichletCondition::dofs() const noexcept
{
  return dofs_;
}

const Vector<Real>& DirichletCondition::vals() const noexcept
{
  return vals_;
}

void DirichletCondition::apply(CsrMatrix& A, Vector<Real>& b) const
{
  if (dofs_.size() != vals_.size())
  {
    throw std::runtime_error("DirichletCondition has inconsistent data");
  }

  const Index* rp   = A.rowPtrData();
  const Index* ci   = A.colIndData();
  Real*        vals = A.valuesData();

  Vector<char> is_dirichlet(A.rows(), 0);
  Vector<char> found_diagonal(A.rows(), 0);
  Vector<Real> dirichlet_values(A.rows());

  for (Index c = 0; c < dofs_.size(); ++c)
  {
    const Index id    = dofs_[c];
    const Real  value = vals_[c];

    if (id < 0 || id >= A.rows() || id >= b.size())
    {
      throw std::runtime_error("Dirichlet id is out of range");
    }

    is_dirichlet[id]     = 1;
    dirichlet_values[id] = value;
  }

  for (Index row = 0; row < A.rows(); ++row)
  {
    const bool row_is_dirichlet = is_dirichlet[row] != 0;

    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      const Index col = ci[k];

      if (row_is_dirichlet)
      {
        vals[k] = 0.0;
        if (col == row)
        {
          vals[k]             = 1.0;
          found_diagonal[row] = 1;
        }
      }
      else if (is_dirichlet[col] != 0)
      {
        b[row]  -= vals[k] * dirichlet_values[col];
        vals[k]  = 0.0;
      }
    }

    if (row_is_dirichlet)
    {
      b[row] = dirichlet_values[row];
    }
  }

  for (Index id = 0; id < A.rows(); ++id)
  {
    if (is_dirichlet[id] != 0 && found_diagonal[id] == 0)
    {
      throw std::runtime_error("Dirichlet row has no diagonal entry");
    }
  }
}

} // namespace femx
