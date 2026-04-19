#pragma once

#include <functional>
#include <vector>

#include <refem/common/Types.hpp>
#include <refem/mesh/Mesh.hpp>

namespace refem
{

class FESpace;
class SparseMatrix;
class Vector;

class DirichletCondition
{
public:
  using ValueFunction = std::function<real_type(const Mesh::Node&)>;

  DirichletCondition() = default;

  static DirichletCondition onBoundary(const FESpace& space,
                                       real_type      value = 0.0);

  static DirichletCondition onBoundary(const FESpace&     space,
                                       const ValueFunction& value);

  void addDof(index_type dof, real_type value);

  const std::vector<index_type>& dofs() const noexcept;
  const std::vector<real_type>&  values() const noexcept;

  void apply(SparseMatrix& A, Vector& b) const;

private:
  std::vector<index_type> dofs_;
  std::vector<real_type>  values_;
};

} // namespace refem
