#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace femx
{

/** @brief Enforce a runtime precondition in every build configuration. */
inline void require(bool cond, std::string_view msg)
{
  if (!cond)
  {
    throw std::runtime_error(std::string(msg));
  }
}

} // namespace femx
