#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>

int main()
{
  femx::linalg::ReSolveOptions opts;
  return opts.max_its > 0 ? 0 : 1;
}
