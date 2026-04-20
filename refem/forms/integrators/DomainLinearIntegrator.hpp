#pragma once

#include <refem/common/Types.hpp>

namespace refem
{
class ElementValues;
class Vector;

class DomainLinearIntegrator
{
public:
  virtual ~DomainLinearIntegrator() = default;

  virtual void assemble(const ElementValues& ev,
                        Vector&              Fe) const = 0;
};
} // namespace refem
