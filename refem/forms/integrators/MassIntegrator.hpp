#pragma once

#include <refem/common/Types.hpp>
#include <refem/forms/integrators/DomainBilinearIntegrator.hpp>

namespace refem
{

class MassIntegrator : public DomainBilinearIntegrator
{
public:
  explicit MassIntegrator(real_type coefficient = 1.0);

  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;

private:
  real_type coeff_{1.0};
};

} // namespace refem
