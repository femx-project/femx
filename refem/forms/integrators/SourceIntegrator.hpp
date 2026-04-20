#pragma once

#include <refem/common/Types.hpp>
#include <refem/forms/integrators/DomainLinearIntegrator.hpp>

namespace refem
{

class SourceIntegrator : public DomainLinearIntegrator
{
public:
  explicit SourceIntegrator(real_type value);

  void assemble(const ElementValues& ev,
                Vector&              Fe) const override;

private:
  real_type value_{0.0};
};

} // namespace refem
