#pragma once

// Compatibility objective interface. Prefer <femx/problem/Objective.hpp>.
#include <femx/problem/Objective.hpp>

namespace femx
{
namespace problem
{

/** @brief Scalar objective functional J(u, m). */
class ObjectiveFunctional : public problem::Objective
{
public:
  ~ObjectiveFunctional() override = default;
};

} // namespace problem
} // namespace femx
