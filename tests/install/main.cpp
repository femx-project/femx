#include <cmath>
#include <exception>
#include <iostream>
#include <string>

#include <femx/fem/Mesh.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/DenseLinearSolver.hpp>

namespace
{

void checkClose(femx::Real actual, femx::Real expected, const char* label)
{
  if (std::abs(actual - expected) > 1.0e-12)
  {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

} // namespace

int main()
{
  try
  {
    femx::fem::Mesh mesh;
    (void) mesh;
    femx::io::VtuWriter writer;
    (void) writer;

    femx::DenseMatrix A;
    A.resize(2, 2);
    A.setZero();
    A.set(0, 0, 3.0);
    A.set(0, 1, 1.0);
    A.set(1, 0, 1.0);
    A.set(1, 1, 2.0);

    femx::HostVector rhs(2);
    rhs[0] = 5.0;
    rhs[1] = 5.0;

    femx::linalg::DenseLinearSolver solver;
    femx::HostVector                x;
    solver.solve(A, rhs, x);

    checkClose(x[0], 1.0, "x[0]");
    checkClose(x[1], 2.0, "x[1]");

    femx::HostVector Ax;
    A.apply(x, Ax);
    checkClose(Ax[0], rhs[0], "Ax[0]");
    checkClose(Ax[1], rhs[1], "Ax[1]");
  }
  catch (const std::exception& e)
  {
    std::cerr << "femx install test failed: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
