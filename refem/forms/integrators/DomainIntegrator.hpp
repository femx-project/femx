#pragma once

#include <refem/common/Types.hpp>

namespace refem
{

class DenseMatrix;
class ElementValues;
class FiniteElement;

class DomainIntegrator
{
public:
  virtual ~DomainIntegrator() = default;

  virtual void assemble(const ElementValues& values,
                        DenseMatrix&         Ke) const = 0;
};

} // namespace refem
