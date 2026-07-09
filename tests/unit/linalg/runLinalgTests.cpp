#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
#include <femx/linalg/native/DenseAssemblyMatrix.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(Real a, Real b)
{
  return std::abs(a - b) <= 1.0e-12;
}

template <class T, std::size_t N>
bool valuesEqual(const T* actual, const std::array<T, N>& expected)
{
  for (std::size_t i = 0; i < N; ++i)
  {
    if (actual[i] != expected[i])
    {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool valuesNear(const Real* actual, const std::array<Real, N>& expected)
{
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[i], expected[i]))
    {
      return false;
    }
  }
  return true;
}

CsrPattern makeSharedElementPattern()
{
  return CsrPattern(3,
                    3,
                    2,
                    [](Index ie, Vector<Index>& dofs)
                    {
                      if (ie == 0)
                      {
                        dofs = {0, 1};
                      }
                      else
                      {
                        dofs = {1, 2};
                      }
                    });
}

DenseMatrix makeLocalMatrix(Real a00, Real a01, Real a10, Real a11)
{
  DenseMatrix local(2, 2);
  local(0, 0) = a00;
  local(0, 1) = a01;
  local(1, 0) = a10;
  local(1, 1) = a11;
  return local;
}

TestOutcome vectorBasics()
{
  TestStatus status(__func__);

  Vector<Real> v(3, 2.0);
  status *= v.size() == 3;
  status *= near(v[0], 2.0) && near(v[1], 2.0) && near(v[2], 2.0);

  v       = {1.0, 2.0, 3.0};
  status *= v.size() == 3;
  status *= near(v.front(), 1.0) && near(v.back(), 3.0);

  v.setZero();
  status *= valuesNear(v.data(), std::array<Real, 3>{{0.0, 0.0, 0.0}});

  resizeOrZero(v, 3);
  status *= v.size() == 3;
  status *= valuesNear(v.data(), std::array<Real, 3>{{0.0, 0.0, 0.0}});

  resizeOrZero(v, 5);
  status *= v.size() == 5;
  status *= valuesNear(v.data(), std::array<Real, 5>{{0.0, 0.0, 0.0, 0.0, 0.0}});

  bool threw = false;
  try
  {
    Vector<Real> invalid(-1);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome vectorViewCopiesAndAssigns()
{
  TestStatus status(__func__);

  Real             raw[3] = {1.0, 2.0, 3.0};
  VectorView<Real> view(raw, 3);
  Vector<Real>     copied(view);

  status *= copied.size() == 3;
  status *= valuesNear(copied.data(), std::array<Real, 3>{{1.0, 2.0, 3.0}});

  Vector<Real> source{4.0, 5.0, 6.0};
  view    = source;
  status *= valuesNear(raw, std::array<Real, 3>{{4.0, 5.0, 6.0}});

  view.setZero();
  status *= valuesNear(raw, std::array<Real, 3>{{0.0, 0.0, 0.0}});

  bool threw = false;
  try
  {
    Vector<Real> wrong_size{1.0, 2.0};
    view = wrong_size;
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome denseMatrixBasics()
{
  TestStatus status(__func__);

  DenseMatrix empty;
  status *= empty.rows() == 0;
  status *= empty.cols() == 0;
  status *= empty.size() == 0;

  DenseMatrix matrix(2, 3);
  status *= matrix.rows() == 2;
  status *= matrix.cols() == 3;
  status *= matrix.size() == 6;

  matrix(0, 0) = 1.0;
  matrix(0, 1) = 2.0;
  matrix(0, 2) = 3.0;
  matrix(1, 0) = 4.0;
  matrix(1, 1) = 5.0;
  matrix(1, 2) = 6.0;

  status *= near(matrix(1, 2), 6.0);
  status *= valuesNear(matrix.data(),
                       std::array<Real, 6>{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}});

  const Vector<Real> x{1.0, 2.0, 3.0};
  Vector<Real>       y;
  matrix.matvec(x, y);
  status *= y.size() == 2;
  status *= valuesNear(y.data(), std::array<Real, 2>{{14.0, 32.0}});

  const Vector<Real> xt{2.0, -1.0};
  Vector<Real>       yt;
  matrix.matvecT(xt, yt);
  status *= yt.size() == 3;
  status *= valuesNear(yt.data(), std::array<Real, 3>{{-2.0, -1.0, 0.0}});

  matrix.setZero();
  status *= valuesNear(matrix.data(),
                       std::array<Real, 6>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  matrix.resize(3, 2);
  status *= matrix.rows() == 3;
  status *= matrix.cols() == 2;
  status *= matrix.size() == 6;
  status *= valuesNear(matrix.data(),
                       std::array<Real, 6>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  return status.report();
}

TestOutcome denseAssemblyMatrixMatvecAliasesApply()
{
  TestStatus status(__func__);

  linalg::DenseAssemblyMatrix matrix;
  matrix.resize(2, 3);
  matrix.set(0, 0, 1.0);
  matrix.set(0, 1, 2.0);
  matrix.set(0, 2, 3.0);
  matrix.set(1, 0, 4.0);
  matrix.set(1, 1, 5.0);
  matrix.set(1, 2, 6.0);

  const Vector<Real> x{1.0, 2.0, 3.0};

  Vector<Real> y;
  matrix.matvec(x, y);
  status *= y.size() == 2;
  status *= valuesNear(y.data(), std::array<Real, 2>{{14.0, 32.0}});

  const Vector<Real> xt{2.0, -1.0};
  Vector<Real>       yt;
  matrix.matvecT(xt, yt);
  status *= yt.size() == 3;
  status *= valuesNear(yt.data(), std::array<Real, 3>{{-2.0, -1.0, 0.0}});

  bool threw = false;
  try
  {
    Vector<Real> wrong_input(2);
    matrix.matvec(wrong_input, y);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome denseAssemblyMatrixBuildsAndClearsValues()
{
  TestStatus status(__func__);

  linalg::DenseAssemblyMatrix matrix;
  matrix.resize(2, 2);
  status *= matrix.numRows() == 2;
  status *= matrix.numCols() == 2;

  matrix.set(0, 0, 1.0);
  matrix.set(0, 1, 2.0);
  matrix.set(1, 0, 3.0);
  matrix.set(1, 1, 4.0);
  matrix.add(0, 1, 5.0);
  matrix.addAtomic(1, 0, 0.5);
  matrix.finalize();

  status *= valuesNear(matrix.mat().data(),
                       std::array<Real, 4>{{1.0, 7.0, 3.5, 4.0}});

  matrix.setZero();
  status *= valuesNear(matrix.mat().data(),
                       std::array<Real, 4>{{0.0, 0.0, 0.0, 0.0}});

  return status.report();
}

TestOutcome csrPatternBuildsSharedElementSparsity()
{
  TestStatus status(__func__);

  CsrPattern pattern = makeSharedElementPattern();

  status *= pattern.rows() == 3;
  status *= pattern.cols() == 3;
  status *= pattern.numElems() == 2;
  status *= pattern.numCooEntries() == 8;
  status *= pattern.nnz() == 7;

  status *= valuesEqual(pattern.rowPtrData(), std::array<Index, 4>{{0, 2, 5, 7}});
  status *= valuesEqual(pattern.colIndData(), std::array<Index, 7>{{0, 1, 0, 1, 2, 1, 2}});
  status *= valuesEqual(pattern.elemCooOffsetData(), std::array<Index, 3>{{0, 4, 8}});
  status *= valuesEqual(pattern.cellNumDofsData(), std::array<Index, 2>{{2, 2}});

  const std::array<Index, 8> expected_map{{0, 1, 2, 3, 3, 4, 5, 6}};
  for (Index i = 0; i < 8; ++i)
  {
    status *= pattern.mapToCsr(i) == expected_map[static_cast<std::size_t>(i)];
  }

  status *= pattern.elemCooOffset(0) == 0;
  status *= pattern.elemCooOffset(1) == 4;
  status *= pattern.elemNumDofs(0) == 2;
  status *= pattern.elemNumDofs(1) == 2;

  return status.report();
}

TestOutcome csrMatrixOwnsValuesForPattern()
{
  TestStatus status(__func__);

  CsrPattern pattern = makeSharedElementPattern();

  CsrMatrix matrix(pattern);

  status *= &matrix.pattern() == &pattern;
  status *= matrix.rows() == 3;
  status *= matrix.cols() == 3;
  status *= matrix.nnz() == 7;
  status *= valuesEqual(matrix.rowPtrData(), std::array<Index, 4>{{0, 2, 5, 7}});
  status *= valuesEqual(matrix.colIndData(), std::array<Index, 7>{{0, 1, 0, 1, 2, 1, 2}});
  status *= valuesNear(matrix.valuesData(),
                       std::array<Real, 7>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  matrix.valuesData()[0]  = 1.0;
  matrix.valuesData()[3]  = 2.0;
  matrix.valuesData()[6]  = 3.0;
  status                 *= valuesNear(matrix.valuesData(),
                       std::array<Real, 7>{{1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0}});

  matrix.setZero();
  status *= valuesNear(matrix.valuesData(),
                       std::array<Real, 7>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  return status.report();
}

TestOutcome csrAssemblyMatrixBuildsAndClearsValues()
{
  TestStatus status(__func__);

  CsrPattern                pattern = makeSharedElementPattern();
  linalg::CsrAssemblyMatrix matrix(pattern);

  matrix.set(0, 0, 1.0);
  matrix.set(0, 1, 2.0);
  matrix.set(1, 0, 3.0);
  matrix.set(1, 1, 4.0);
  matrix.set(1, 2, 5.0);
  matrix.set(2, 1, 6.0);
  matrix.set(2, 2, 7.0);
  matrix.add(1, 1, 0.5);
  matrix.addAtomic(2, 2, 0.25);
  matrix.finalize();

  status *= valuesNear(matrix.mat().valuesData(),
                       std::array<Real, 7>{{1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.25}});

  bool resize_threw = false;
  try
  {
    matrix.resize(2, 3);
  }
  catch (const std::runtime_error&)
  {
    resize_threw = true;
  }
  status *= resize_threw;

  bool entry_threw = false;
  try
  {
    matrix.set(0, 2, 1.0);
  }
  catch (const std::runtime_error&)
  {
    entry_threw = true;
  }
  status *= entry_threw;

  matrix.setZero();
  status *= valuesNear(matrix.mat().valuesData(),
                       std::array<Real, 7>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  return status.report();
}

TestOutcome csrAssemblyMatrixAddsMappedElementMatrices()
{
  TestStatus status(__func__);

  CsrPattern                pattern = makeSharedElementPattern();
  linalg::CsrAssemblyMatrix matrix(pattern);
  const DenseMatrix         elem0 = makeLocalMatrix(1.0, 2.0, 3.0, 4.0);
  const DenseMatrix         elem1 = makeLocalMatrix(5.0, 6.0, 7.0, 8.0);

  status *= matrix.addMappedMat(0, elem0, false);
  status *= matrix.addMappedMat(1, elem1, false);
  status *= valuesNear(matrix.mat().valuesData(),
                       std::array<Real, 7>{{1.0, 2.0, 3.0, 9.0, 6.0, 7.0, 8.0}});

  matrix.setZero();
  status *= matrix.addMappedMat(0, elem0, true);
  status *= matrix.addMappedMat(1, elem1, true);
  status *= valuesNear(matrix.mat().valuesData(),
                       std::array<Real, 7>{{1.0, 2.0, 3.0, 9.0, 6.0, 7.0, 8.0}});

  bool threw = false;
  try
  {
    DenseMatrix wrong_size(1, 1);
    matrix.addMappedMat(0, wrong_size, false);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome csrAssemblyMatrixMatvecAliasesApply()
{
  TestStatus status(__func__);

  CsrPattern                pattern = makeSharedElementPattern();
  linalg::CsrAssemblyMatrix matrix(pattern);

  matrix.set(0, 0, 1.0);
  matrix.set(0, 1, 2.0);
  matrix.set(1, 0, 3.0);
  matrix.set(1, 1, 4.0);
  matrix.set(1, 2, 5.0);
  matrix.set(2, 1, 6.0);
  matrix.set(2, 2, 7.0);

  const Vector<Real> x{1.0, 2.0, 3.0};
  Vector<Real>       y;
  matrix.matvec(x, y);
  status *= y.size() == 3;
  status *= valuesNear(y.data(), std::array<Real, 3>{{5.0, 26.0, 33.0}});

  Vector<Real> yt;
  matrix.matvecT(x, yt);
  status *= yt.size() == 3;
  status *= valuesNear(yt.data(), std::array<Real, 3>{{7.0, 28.0, 31.0}});

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::vectorBasics();
  results += femx::tests::vectorViewCopiesAndAssigns();
  results += femx::tests::denseMatrixBasics();
  results += femx::tests::denseAssemblyMatrixMatvecAliasesApply();
  results += femx::tests::denseAssemblyMatrixBuildsAndClearsValues();
  results += femx::tests::csrPatternBuildsSharedElementSparsity();
  results += femx::tests::csrMatrixOwnsValuesForPattern();
  results += femx::tests::csrAssemblyMatrixBuildsAndClearsValues();
  results += femx::tests::csrAssemblyMatrixAddsMappedElementMatrices();
  results += femx::tests::csrAssemblyMatrixMatvecAliasesApply();

  return results.summary();
}
