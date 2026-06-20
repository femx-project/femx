#include <femx/algebra/DenseMatrix.hpp>
#include <femx/algebra/LinearOperator.hpp>
#include <femx/algebra/LinearSolver.hpp>
#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/algebra/MatrixOperator.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/problem/Observation.hpp>

int main()
{
  femx::Vector<femx::Real> x(3);
  femx::DenseMatrix        a(2, 2);

  femx::algebra::LinearOperator* op      = nullptr;
  femx::algebra::LinearSolver*   solver  = nullptr;
  femx::algebra::MatrixBuilder*  builder = nullptr;
  femx::algebra::MatrixOperator* matrix  = nullptr;
  femx::problem::Observation*    obs     = nullptr;
  (void)op;
  (void)solver;
  (void)builder;
  (void)matrix;
  (void)obs;

  return x.size() == 3 && a.rows() == 2 && a.cols() == 2 ? 0 : 1;
}
