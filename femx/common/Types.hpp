#pragma once

#include <cstdint>
#include <limits>

namespace femx
{

using Real  = double;
using Index = std::int32_t;

namespace constants
{
constexpr Real ZERO      = 0.0;
constexpr Real ONE       = 1.0;
constexpr Real TWO       = 2.0;
constexpr Real HALF      = 0.5;
constexpr Real MINUS_ONE = -1.0;

constexpr Real MACHINE_EPSILON = std::numeric_limits<Real>::epsilon();
} // namespace constants

namespace colors
{
// must be const pointer and const dest for
// const string declarations to pass -Wwrite-strings
static const char* const RED    = "\033[1;31m";
static const char* const GREEN  = "\033[1;32m";
static const char* const YELLOW = "\033[33;1m";
static const char* const BLUE   = "\033[34;1m";
static const char* const ORANGE = "\u001b[38;5;208m";
static const char* const CLEAR  = "\033[0m";
} // namespace colors

} // namespace femx
