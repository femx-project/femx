#pragma once

#include <type_traits>

#include <femx/common/Types.hpp>

#if defined(FEMX_HAS_ENZYME)

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_const;
extern int enzyme_out;

template <typename Return, typename Fn, typename... Args>
Return __enzyme_autodiff(Fn fn, Args... args);

#endif

namespace femx
{
namespace ad
{

#if defined(FEMX_HAS_ENZYME)

inline constexpr bool has_enzyme = true;

template <typename Return, typename Fn, typename... Args>
Return autodiff(Fn fn, Args... args)
{
  return __enzyme_autodiff<Return>(fn, args...);
}

template <typename Fn>
real_type derivative(Fn fn, real_type x)
{
  static_assert(std::is_invocable_r_v<real_type, Fn, real_type>,
                "derivative expects a scalar function real_type(real_type).");
  return autodiff<real_type>(fn, x);
}

#else

inline constexpr bool has_enzyme = false;

template <typename>
inline constexpr bool always_false = false;

template <typename Return, typename Fn, typename... Args>
Return autodiff(Fn, Args...)
{
  static_assert(always_false<Return>,
                "femx was built without Enzyme. Configure with "
                "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
}

template <typename Fn>
real_type derivative(Fn, real_type)
{
  static_assert(always_false<Fn>,
                "femx was built without Enzyme. Configure with "
                "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
}

#endif

} // namespace ad
} // namespace femx
