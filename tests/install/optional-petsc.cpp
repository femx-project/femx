#include <femx/linalg/petsc/KspLinearSolver.hpp>

int main()
{
  femx::linalg::KspOptions opts;
  return opts.max_its > 0 ? 0 : 1;
}
