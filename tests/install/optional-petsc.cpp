#include <femx/linalg/petsc/KspLinearSolver.hpp>

int main()
{
  femx::linalg::KspOptions opts;
  return opts.type == KSPGMRES && opts.pc_type == PCILU
                 && opts.factor_levels == 0 && opts.restart == 200
             ? 0
             : 1;
}
