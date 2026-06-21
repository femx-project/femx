#include <iomanip>
#include <iostream>

#include "Problem.hpp"

int main()
{
  using namespace femx;
  using namespace femx::examples_inverse_linear_ctr_new_api;

  LinearControlSetup setup;

  Vector<Real> prm = makeInitialParams();
  Vector<Real> state;
  Vector<Real> grad;

  setup.state_solver.solve(prm, state);
  const Real value = setup.functional.valueGrad(prm, grad);

  std::cout << std::setprecision(12);
  std::cout << "example-inverse-linear-control-new-api\n";
  printVector("prm", prm);
  printVector("state", state);
  std::cout << "value = " << value << '\n';
  printVector("grad", grad);

  return 0;
}
