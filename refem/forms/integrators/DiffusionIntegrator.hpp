#pragma once

#include <refem/common/Types.hpp>
#include <refem/forms/integrators/DomainIntegrator.hpp>

namespace refem
{

class DiffusionIntegrator : public DomainIntegrator
{
public:
  explicit DiffusionIntegrator(real_type kappa)
    : kappa_(kappa)
  {
  }

  void assemble(const ElementValues& values,
                DenseMatrix&         Ke) const override;

private:
  real_type kappa_ = 1.0;
};

} // namespace refem
