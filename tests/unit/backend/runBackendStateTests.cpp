#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "TestHelper.hpp"
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/CsrTranspose.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace tests
{
namespace
{

bool near(Real lhs, Real rhs)
{
  return std::abs(lhs - rhs) <= 1.0e-12;
}

template <class Values, std::size_t N>
bool valsNear(const Values& vals, const std::array<Real, N>& expected)
{
  if (vals.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (Index i = 0; i < vals.size(); ++i)
  {
    if (!near(vals[i], expected[static_cast<std::size_t>(i)]))
    {
      return false;
    }
  }
  return true;
}

HostCsrGraph denseThreeByThreeGraph()
{
  return {3,
          3,
          HostIndexVector{0, 3, 6, 9},
          HostIndexVector{0, 1, 2, 0, 1, 2, 0, 1, 2}};
}

void setMatVals(HostCsrMatrix& mat)
{
  mat.vals() = {4.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0};
}

HostVector mul(const HostCsrMatrix& mat, const HostVector& in)
{
  HostVector out(mat.rows());
  for (Index row = 0; row < mat.rows(); ++row)
  {
    for (Index k = mat.rowPtrData()[row];
         k < mat.rowPtrData()[row + 1];
         ++k)
    {
      out[row] += mat.valsData()[k]
                  * in[mat.colIndData()[k]];
    }
  }
  return out;
}

Real dot(const HostVector& lhs, const HostVector& rhs)
{
  Real result = 0.0;
  for (Index i = 0; i < lhs.size(); ++i)
  {
    result += lhs[i] * rhs[i];
  }
  return result;
}

TestOutcome boundaryRowsAndForwardEliminationStayDistinct()
{
  TestStatus status(__func__);

  const HostCsrGraph graph = denseThreeByThreeGraph();
  const auto         map   = assembly::makeBoundaryMap(Array<Index>{1}, graph);

  HostCsrMatrix authoritative(graph);
  setMatVals(authoritative);
  assembly::replaceRows(map, authoritative);
  status *= valsNear(authoritative.vals(),
                     std::array<Real, 9>{{4.0, 1.0, 2.0, 0.0, 1.0, 0.0, 7.0, 8.0, 9.0}});

  HostVector res{10.0, 20.0, 30.0};
  assembly::replaceRes(map,
                       HostVector{4.0, 7.0, 9.0},
                       HostVector{2.5},
                       res);
  status *= valsNear(res,
                     std::array<Real, 3>{{10.0, 4.5, 30.0}});

  HostCsrMatrix solve_mat(graph);
  setMatVals(solve_mat);
  HostVector rhs{10.0, 20.0, 30.0};
  assembly::prepareForwardSolve(
      map, solve_mat, rhs, HostVector{2.0});
  status *= valsNear(solve_mat.vals(),
                     std::array<Real, 9>{{4.0, 0.0, 2.0, 0.0, 1.0, 0.0, 7.0, 0.0, 9.0}});
  status *= valsNear(rhs, std::array<Real, 3>{{8.0, 2.0, 14.0}});

  return status.report();
}

TestOutcome persistentTransposeSupportsAdjointIdentity()
{
  TestStatus status(__func__);

  const HostCsrGraph graph = denseThreeByThreeGraph();
  HostCsrMatrix      mat(graph);
  setMatVals(mat);

  const HostCsrTransposeMap tr_map(graph);
  HostCsrMatrix             tr(tr_map.trGraph());
  trVals(mat, tr_map, tr);

  const HostVector x{1.0, -2.0, 0.5};
  const HostVector y{0.25, 3.0, -1.0};
  status *= near(dot(mul(mat, x), y),
                 dot(x, mul(tr, y)));

  mat.valsData()[1] = 11.0;
  trVals(mat, tr_map, tr);
  status *= near(dot(mul(mat, x), y),
                 dot(x, mul(tr, y)));

  return status.report();
}

TestOutcome trMatGraphSurvivesMapMove()
{
  TestStatus status(__func__);

  const HostCsrGraph graph = denseThreeByThreeGraph();
  HostCsrMatrix      mat(graph);
  setMatVals(mat);

  HostCsrTransposeMap map(graph);
  HostCsrMatrix       tr(map.trGraph());
  auto                moved_map = std::move(map);
  trVals(mat, moved_map, tr);

  status *= tr.rows() == mat.cols();
  status *= near(tr.valsData()[0], 4.0);
  status *= near(dot(mul(mat, HostVector{1.0, 2.0, 3.0}),
                     HostVector{0.5, -1.0, 2.0}),
                 dot(HostVector{1.0, 2.0, 3.0},
                     mul(tr,
                         HostVector{0.5, -1.0, 2.0})));

  return status.report();
}

TestOutcome trajectoryOwnsContiguousStorage()
{
  TestStatus status(__func__);

  state::TimeTrajectory trajectory(2, 3);
  trajectory.level(0)  = HostVector{1.0, 2.0, 3.0};
  trajectory.level(1)  = HostVector{4.0, 5.0, 6.0};
  trajectory.level(2)  = HostVector{7.0, 8.0, 9.0};
  status              *= trajectory.numTimeLevels() == 3;
  status              *= trajectory.level(1)[2] == 6.0;
  status              *= trajectory.data() + 3 == trajectory.level(1).data();

  return status.report();
}

TestOutcome boundaryRejectsWrongLayoutsAndAliasedResiduals()
{
  TestStatus status(__func__);

  const HostCsrGraph graph = denseThreeByThreeGraph();
  const auto         map   = assembly::makeBoundaryMap(Array<Index>{0, 2}, graph);
  const HostCsrGraph different_layout{
      3,
      3,
      HostIndexVector{0, 3, 6, 9},
      HostIndexVector{1, 0, 2, 0, 2, 1, 2, 1, 0}};
  HostCsrMatrix wrong_mat(different_layout);

  bool layout_rejected = false;
  try
  {
    assembly::replaceRows(map, wrong_mat);
  }
  catch (const std::runtime_error&)
  {
    layout_rejected = true;
  }
  status *= layout_rejected;

  HostVector alias_vec{1.0, 2.0, 3.0};
  bool       alias_rejected = false;
  try
  {
    assembly::replaceRes(map,
                         alias_vec,
                         HostVector{0.0, 0.0},
                         alias_vec);
  }
  catch (const std::runtime_error&)
  {
    alias_rejected = true;
  }
  status *= alias_rejected;

  const HostCsrGraph diagonal_graph{
      3,
      3,
      HostIndexVector{0, 1, 2, 3},
      HostIndexVector{0, 1, 2}};
  const auto diagonal_map =
      assembly::makeBoundaryMap(Array<Index>{0}, diagonal_graph);
  HostCsrMatrix diag_mat(diagonal_graph);
  diag_mat.vals() = {2.0, 3.0, 4.0};

  bool mat_alias_rejected = false;
  try
  {
    assembly::prepareForwardSolve(diagonal_map,
                                  diag_mat,
                                  diag_mat.vals(),
                                  HostVector{1.0});
  }
  catch (const std::runtime_error&)
  {
    mat_alias_rejected = true;
  }
  status *= mat_alias_rejected;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::boundaryRowsAndForwardEliminationStayDistinct();
  results += femx::tests::persistentTransposeSupportsAdjointIdentity();
  results += femx::tests::trMatGraphSurvivesMapMove();
  results += femx::tests::trajectoryOwnsContiguousStorage();
  results += femx::tests::boundaryRejectsWrongLayoutsAndAliasedResiduals();
  return results.summary();
}
