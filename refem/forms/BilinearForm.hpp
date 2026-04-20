#pragma once

#include <memory>
#include <vector>

#include <refem/common/Types.hpp>
#include <refem/linalg/FixedSparsityPattern.hpp>
#include <refem/linalg/SparseMatrix.hpp>

namespace refem
{

class FESpace;
class DomainBilinearIntegrator;

class BilinearForm
{
public:
  explicit BilinearForm(const FESpace* space);

  void addDomainIntegrator(std::unique_ptr<DomainBilinearIntegrator> integrator);
  void assemble();

  const SparseMatrix& matrix() const;
  SparseMatrix&       matrix();

private:
  const FESpace* space_{nullptr};

  FixedSparsityPattern pattern_;
  SparseMatrix         K_;

  std::vector<std::unique_ptr<DomainBilinearIntegrator>> terms_;
};

} // namespace refem
