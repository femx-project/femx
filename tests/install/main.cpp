#include <cmath>
#include <exception>
#include <iostream>
#include <string>

#include <femx/ad/Enzyme.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Vector.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/native/HostLinearSystem.hpp>

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
    (void) femx::ad::has_enzyme;

    femx::HostCsrPattern pattern(2,
                                 2,
                                 femx::HostVector<femx::Index>{0, 2, 4},
                                 femx::HostVector<femx::Index>{0, 1, 0, 1});
    femx::HostCsrMatrix  A(std::move(pattern));
    A.vals() = femx::HostVector<femx::Real>{3.0, 1.0, 1.0, 2.0};

    femx::HostVector<femx::Real> rhs(2);
    rhs[0] = 5.0;
    rhs[1] = 5.0;
    checkClose(femx::norm(rhs), std::sqrt(50.0), "rhs norm");

    femx::linalg::HostLinearSystem system;
    auto&                          jac = system.matrix();
    jac.setup(A.pattern());

    femx::HostVector<femx::Index> rows{0, 1};
    femx::HostVector<femx::Index> columns{0, 1};
    femx::HostVector<femx::Index> entries{0, 1, 2, 3};
    femx::DenseMatrix             values(2, 2);
    values(0, 0) = 3.0;
    values(0, 1) = 1.0;
    values(1, 0) = 1.0;
    values(1, 1) = 2.0;
    jac.addElement(
        {rows.view(), columns.view(), entries.view(), values.view()});
    jac.finalize();

    femx::HostVector<femx::Real> x;
    system.solve(rhs.view(), x);

    checkClose(x[0], 1.0, "x[0]");
    checkClose(x[1], 2.0, "x[1]");

    femx::HostVector<femx::Real> Ax;
    jac.apply(x.view(), Ax);
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
