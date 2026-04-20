#pragma once

#include <vector>

#include <refem/common/Types.hpp>
#include <refem/forms/integrators/DomainBilinearIntegrator.hpp>

namespace refem
{

class AdvectionIntegrator : public DomainBilinearIntegrator
{
public:
  AdvectionIntegrator(std::vector<real_type> velocity,
                      real_type              coefficient = 1.0);

  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;

private:
  std::vector<real_type> velocity_;
  real_type              coeff_{1.0};
};

} // namespace refem
