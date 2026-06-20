#pragma once

// Compatibility matrix type. Prefer <femx/algebra/MatrixOperator.hpp>.
#include <femx/algebra/MatrixOperator.hpp>

namespace femx
{
namespace algebra
{

/** @brief Compatibility matrix type. Prefer algebra::MatrixOperator. */
class SystemMatrix : public algebra::MatrixOperator
{
public:
  ~SystemMatrix() override = default;
};

} // namespace algebra
} // namespace femx
