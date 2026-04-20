#pragma once

#include <refem/common/Types.hpp>

namespace refem
{

class DenseMatrix;
class ElementValues;

class DomainBilinearIntegrator
{
public:
  virtual ~DomainBilinearIntegrator() = default;

  virtual void assemble(const ElementValues& ev,
                        DenseMatrix&         Ke) const = 0;
};

} // namespace refem
