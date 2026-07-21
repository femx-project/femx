#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

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
bool valsEqual(const T* actual, const std::array<T, N>& expected)
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
bool valsNear(const Real* actual, const std::array<Real, N>& expected)
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

assembly::HostAssemblyMap makeSharedElementMap()
{
  const Array<Array<Index>> dofs{{0, 1}, {1, 2}};
  return assembly::makeAssemblyMap(3, 3, dofs, dofs);
}

TestOutcome vectorBasics()
{
  TestStatus                status(__func__);
  CpuContext                ctx;
  linalg::HostVectorHandler vec_handler(ctx);

  HostVector v(3, 2.0);
  status *= v.size() == 3;
  status *= near(v[0], 2.0) && near(v[1], 2.0) && near(v[2], 2.0);

  v       = {1.0, 2.0, 3.0};
  status *= v.size() == 3;
  status *= near(v.front(), 1.0) && near(v.back(), 3.0);

  vec_handler.zero(v.view());
  status *= valsNear(v.data(), std::array<Real, 3>{{0.0, 0.0, 0.0}});

  vec_handler.resizeOrZero(v, 3);
  status *= v.size() == 3;
  status *= valsNear(v.data(), std::array<Real, 3>{{0.0, 0.0, 0.0}});

  vec_handler.resizeOrZero(v, 5);
  status *= v.size() == 5;
  status *= valsNear(v.data(), std::array<Real, 5>{{0.0, 0.0, 0.0, 0.0, 0.0}});

  bool threw = false;
  try
  {
    HostVector invalid(-1);
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
  TestStatus                status(__func__);
  CpuContext                ctx;
  linalg::HostVectorHandler vec_handler(ctx);

  Real           raw[3] = {1.0, 2.0, 3.0};
  HostVectorView view(raw, 3);
  HostVector     copied(view);

  status *= copied.size() == 3;
  status *= valsNear(copied.data(), std::array<Real, 3>{{1.0, 2.0, 3.0}});

  HostVector src{4.0, 5.0, 6.0};
  view    = src;
  status *= valsNear(raw, std::array<Real, 3>{{4.0, 5.0, 6.0}});

  vec_handler.zero(view);
  status *= valsNear(raw, std::array<Real, 3>{{0.0, 0.0, 0.0}});

  bool threw = false;
  try
  {
    HostVector wrong_size{1.0, 2.0};
    view = wrong_size;
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome vectorGatherScatter()
{
  TestStatus status(__func__);

  const HostVector          source{10.0, 20.0, 30.0, 40.0, 50.0};
  const HostIndexVector     indices{4, 1, 3};
  HostVector                compact(3);
  CpuContext                ctx;
  linalg::HostVectorHandler vec_handler(ctx);
  vec_handler.gather(source.view(), indices.view(), compact.view());
  status *= valsNear(compact.data(),
                     std::array<Real, 3>{{50.0, 20.0, 40.0}});

  HostVector expanded(5, -1.0);
  vec_handler.scatter(compact.view(), indices.view(), expanded.view());
  status *= valsNear(expanded.data(),
                     std::array<Real, 5>{{-1.0, 20.0, -1.0, 40.0, 50.0}});

  return status.report();
}

TestOutcome denseMatrixBasics()
{
  TestStatus                status(__func__);
  CpuContext                ctx;
  linalg::HostMatrixHandler mat_handler(ctx);

  DenseMatrix empty;
  status *= empty.rows() == 0;
  status *= empty.cols() == 0;
  status *= empty.size() == 0;

  DenseMatrix mat(2, 3);
  status *= mat.rows() == 2;
  status *= mat.cols() == 3;
  status *= mat.size() == 6;

  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(0, 2) = 3.0;
  mat(1, 0) = 4.0;
  mat(1, 1) = 5.0;
  mat(1, 2) = 6.0;

  status *= near(mat(1, 2), 6.0);
  status *= valsNear(mat.data(),
                     std::array<Real, 6>{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}});

  mat_handler.zero(mat);
  status *= valsNear(mat.data(),
                     std::array<Real, 6>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  mat.resize(3, 2);
  status *= mat.rows() == 3;
  status *= mat.cols() == 2;
  status *= mat.size() == 6;
  status *= valsNear(mat.data(),
                     std::array<Real, 6>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  return status.report();
}

TestOutcome denseMatrixApplies()
{
  TestStatus status(__func__);

  DenseMatrix mat;
  mat.resize(2, 3);
  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(0, 2) = 3.0;
  mat(1, 0) = 4.0;
  mat(1, 1) = 5.0;
  mat(1, 2) = 6.0;

  const HostVector          x{1.0, 2.0, 3.0};
  CpuContext                ctx;
  linalg::HostMatrixHandler mat_handler(ctx);

  HostVector y(2);
  mat_handler.matvec(mat.view(), x.view(), y.view());
  status *= valsNear(y.data(), std::array<Real, 2>{{14.0, 32.0}});

  const HostVector xt{2.0, -1.0};
  HostVector       yt(3);
  mat_handler.matvecT(mat.view(), xt.view(), yt.view());
  status *= valsNear(yt.data(), std::array<Real, 3>{{-2.0, -1.0, 0.0}});

  bool threw = false;
  try
  {
    HostVector wrong_input(2);
    mat_handler.matvec(mat.view(), wrong_input.view(), y.view());
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome assemblyMapBuildsSharedElementSparsity()
{
  TestStatus status(__func__);

  const auto map  = makeSharedElementMap();
  const auto view = map.view();

  status *= map.pattern().rows() == 3;
  status *= map.pattern().cols() == 3;
  status *= map.numElems() == 2;
  status *= view.jac_offsets[map.numElems()] == 8;
  status *= map.pattern().nnz() == 7;

  status *= valsEqual(map.pattern().rowPtrData(),
                      std::array<Index, 4>{{0, 2, 5, 7}});
  status *= valsEqual(map.pattern().colIndData(),
                      std::array<Index, 7>{{0, 1, 0, 1, 2, 1, 2}});
  status *= valsEqual(view.jac_offsets,
                      std::array<Index, 3>{{0, 4, 8}});
  status *= valsEqual(view.res_offsets,
                      std::array<Index, 3>{{0, 2, 4}});

  const std::array<Index, 8> expected_map{{0, 1, 2, 3, 3, 4, 5, 6}};
  for (Index i = 0; i < 8; ++i)
  {
    status *= view.jac_map[i]
              == expected_map[static_cast<std::size_t>(i)];
  }

  status *= view.numResDofs(0) == 2;
  status *= view.numResDofs(1) == 2;
  status *= view.numStateDofs(0) == 2;
  status *= view.numStateDofs(1) == 2;

  return status.report();
}

TestOutcome csrMatrixOwnsValuesForGraph()
{
  TestStatus                status(__func__);
  CpuContext                ctx;
  linalg::HostMatrixHandler mat_handler(ctx);

  const auto    map = makeSharedElementMap();
  HostCsrMatrix mat(map.pattern());

  status *= mat.rows() == 3;
  status *= mat.cols() == 3;
  status *= mat.nnz() == 7;
  status *= valsEqual(mat.rowPtrData(), std::array<Index, 4>{{0, 2, 5, 7}});
  status *= valsEqual(mat.colIndData(), std::array<Index, 7>{{0, 1, 0, 1, 2, 1, 2}});
  status *= valsNear(mat.valsData(),
                     std::array<Real, 7>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  mat.valsData()[0]  = 1.0;
  mat.valsData()[3]  = 2.0;
  mat.valsData()[6]  = 3.0;
  status            *= valsNear(mat.valsData(),
                     std::array<Real, 7>{{1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0}});

  HostCsrMatrix copied(map.pattern());
  mat_handler.copy(mat, copied);
  status *= valsNear(copied.valsData(),
                     std::array<Real, 7>{{1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0}});

  mat_handler.zero(mat);
  status *= valsNear(mat.valsData(),
                     std::array<Real, 7>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  return status.report();
}

TestOutcome csrPatternUsesLayoutIdentity()
{
  TestStatus status(__func__);

  const HostCsrPattern pattern{
      2,
      2,
      HostIndexVector{0, 1, 2},
      HostIndexVector{0, 1}};
  const HostCsrPattern shared = pattern;
  const HostCsrPattern separate{
      2,
      2,
      HostIndexVector{0, 1, 2},
      HostIndexVector{0, 1}};

  status *= pattern.layoutId() != 0;
  status *= shared.layoutId() == pattern.layoutId();
  status *= separate.layoutId() != pattern.layoutId();

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
  results += femx::tests::vectorGatherScatter();
  results += femx::tests::denseMatrixBasics();
  results += femx::tests::denseMatrixApplies();
  results += femx::tests::assemblyMapBuildsSharedElementSparsity();
  results += femx::tests::csrMatrixOwnsValuesForGraph();
  results += femx::tests::csrPatternUsesLayoutIdentity();

  return results.summary();
}
