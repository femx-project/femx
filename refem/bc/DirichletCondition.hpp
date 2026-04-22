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
  using Function = std::function<real_type(const Mesh::Node&, real_type)>;

  DirichletCondition() = default;

  static DirichletCondition onBoundary(const FESpace& space,
                                       real_type      value = 0.0);

  static DirichletCondition onBoundary(const FESpace&       space,
                                       const Function& value,
                                       real_type            time = 0.0);

  void addDof(index_type dof, real_type value);

  const std::vector<index_type>& dofs() const noexcept;
  const std::vector<real_type>&  values() const noexcept;

  void apply(SparseMatrix& A, Vector& b) const;

private:
  std::vector<index_type> dofs_;
  std::vector<real_type>  values_;
};

} // namespace refem
