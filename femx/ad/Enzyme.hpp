#pragma once

#include <femx/core/Types.hpp>

#if defined(FEMX_HAS_ENZYME)

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_const;
extern int enzyme_out;

template <typename Return, typename... Args>
Return __enzyme_autodiff(void* fn, Args... args);

#endif

namespace femx
{
namespace ad
{

#if defined(FEMX_HAS_ENZYME)

inline constexpr bool has_enzyme = true;

template <auto Fn, typename Return, typename... Args>
Return autodiff(Args... args)
{
  return __enzyme_autodiff<Return>(reinterpret_cast<void*>(Fn), args...);
}

template <Real (*Fn)(Real)>
Real derivative(Real x)
{
  return autodiff<Fn, Real>(x);
}

#else

inline constexpr bool has_enzyme = false;

template <typename>
inline constexpr bool always_false = false;

template <auto, typename Return, typename... Args>
Return autodiff(Args...)
{
  static_assert(always_false<Return>,
                "femx was built without Enzyme. Configure with "
                "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
}

template <Real (*)(Real)>
Real derivative(Real)
{
  static_assert(always_false<Real>,
                "femx was built without Enzyme. Configure with "
                "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
}

#endif

} // namespace ad
} // namespace femx
