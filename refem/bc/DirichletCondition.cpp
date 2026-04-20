#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <refem/bc/DirichletCondition.hpp>
#include <refem/fe/FESpace.hpp>
#include <refem/linalg/SparseMatrix.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{
namespace
{

bool isBoundaryNode(const Mesh&     mesh,
                    index_type      node_id,
                    const real_type min_coord[3],
                    const real_type max_coord[3])
{
  constexpr real_type eps = 1.0e-12;

  const auto& point = mesh.node(node_id);
  for (index_type d = 0; d < mesh.dim(); ++d)
  {
    const real_type scale =
        std::max<real_type>(1.0, std::abs(max_coord[d] - min_coord[d]));
    const real_type tol = eps * scale;
    if (std::abs(point[d] - min_coord[d]) <= tol || std::abs(point[d] - max_coord[d]) <= tol)
    {
      return true;
    }
  }

  return false;
}

} // namespace

DirichletCondition DirichletCondition::onBoundary(const FESpace& space,
                                                  real_type      value)
{
  return onBoundary(space,
                    [value](const Mesh::Node&)
                    { return value; });
}

DirichletCondition DirichletCondition::onBoundary(
    const FESpace&       space,
    const ValueFunction& value)
{
  const Mesh& mesh = space.mesh();

  real_type min_coord[3] = {mesh.node(0)[0], mesh.node(0)[1], mesh.node(0)[2]};
  real_type max_coord[3] = {mesh.node(0)[0], mesh.node(0)[1], mesh.node(0)[2]};

  for (index_type in = 1; in < mesh.numNodes(); ++in)
  {
    for (index_type d = 0; d < mesh.dim(); ++d)
    {
      min_coord[d] = std::min(min_coord[d], mesh.node(in)[d]);
      max_coord[d] = std::max(max_coord[d], mesh.node(in)[d]);
    }
  }

  DirichletCondition condition;
  for (index_type in = 0; in < mesh.numNodes(); ++in)
  {
    if (isBoundaryNode(mesh, in, min_coord, max_coord))
    {
      condition.addDof(in, value(mesh.node(in)));
    }
  }

  return condition;
}

void DirichletCondition::addDof(index_type dof, real_type value)
{
  dofs_.push_back(dof);
  values_.push_back(value);
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

  for (std::size_t c = 0; c < dofs_.size(); ++c)
  {
    const index_type dof   = dofs_[c];
    const real_type  value = values_[c];

    if (dof < 0 || dof >= A.rows() || dof >= b.size())
    {
      throw std::runtime_error("Dirichlet dof is out of range");
    }

    for (index_type row = 0; row < A.rows(); ++row)
    {
      for (index_type k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
      {
        if (col_ind[k] == dof)
        {
          if (row != dof)
          {
            b[row]    -= values[k] * value;
            values[k]  = 0.0;
          }
        }
      }
    }

    bool found_diagonal = false;
    for (index_type k = row_ptr[dof]; k < row_ptr[dof + 1]; ++k)
    {
      values[k] = 0.0;
      if (col_ind[k] == dof)
      {
        values[k]      = 1.0;
        found_diagonal = true;
      }
    }

    if (!found_diagonal)
    {
      throw std::runtime_error("Dirichlet row has no diagonal entry");
    }

    b[dof] = value;
  }
}

} // namespace refem
