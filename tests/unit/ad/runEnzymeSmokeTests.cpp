#include <cmath>
#include <iostream>

#include <femx/ad/Enzyme.hpp>
#include <femx/common/Types.hpp>

namespace
{

femx::real_type polynomial(femx::real_type x)
{
  return x * x * x + femx::constants::TWO * x;
}

} // namespace

int main(int, char**)
{
  const femx::real_type x        = 3.0;
  const femx::real_type expected = 3.0 * x * x + femx::constants::TWO;
  const femx::real_type actual   = femx::ad::derivative(polynomial, x);

  if (std::abs(actual - expected) > 1.0e-10)
  {
    std::cerr << "Incorrect Enzyme derivative: expected " << expected << ", got " << actual
              << '\n';
    return 1;
  }

  return 0;
}
