#pragma once

#include <memory>
#include <vector>

#include <refem/common/Types.hpp>
#include <refem/linalg/Vector.hpp>

namespace refem
{

class FESpace;
class DomainLinearIntegrator;

class LinearForm
{
public:
  explicit LinearForm(const FESpace* space);

  void addDomainIntegrator(
      std::unique_ptr<DomainLinearIntegrator> integrator);

  void assemble();

  const Vector& vector() const;
  Vector&       vector();

private:
  const FESpace* space_{nullptr};

  Vector b_;

  std::vector<std::unique_ptr<DomainLinearIntegrator>> terms_;
};

} // namespace refem