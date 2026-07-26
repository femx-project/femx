#include <array>
#include <cmath>

#include "TestHelper.hpp"
#include <femx/assembly/AssemblyMap.hpp>
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostJacobian.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

class IdentityTimeResidual final : public state::HostTimeResidual
{
public:
  IdentityTimeResidual()
    : pattern_(assembly::makeAssemblyMap(
                   3,
                   3,
                   HostVector<HostVector<Index>>{{0, 1, 2}},
                   HostVector<HostVector<Index>>{{0, 1, 2}})
                   .pattern())
  {
  }

  state::TimeDims dims() const override
  {
    return {2, 3, 0, 3, 1};
  }

  const HostCsrPattern& hostPattern() const override
  {
    return pattern_;
  }

  void initialState(HostVectorView<const Real>          prm,
                    HostVector<Real>&                   out,
                    linalg::Context<MemorySpace::Host>& ctx) const override
  {
    require(prm.empty(), "Identity residual is parameter-free");
    ctx.vectors().resizeOrZero(out, 3);
  }

  void applyJacT(const state::HostTimeContext&,
                 state::VariableBlock                wrt,
                 HostVectorView<const Real>          adj,
                 HostVector<Real>&                   out,
                 linalg::Context<MemorySpace::Host>& ctx) const override
  {
    require(!wrt.isNextState(),
            "Identity transpose apply supports history and parameters");
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    ctx.vectors().resizeOrZero(out, 3);
  }

  void assembleNext(const state::HostTimeContext&        time,
                    HostVector<Real>&                    res,
                    linalg::Jacobian<MemorySpace::Host>& out,
                    linalg::Context<MemorySpace::Host>&) const override
  {
    res = time.nxt;
    const HostVector<Index> rows{0, 1, 2};
    const HostVector<Index> entries{0, 1, 2, 3, 4, 5, 6, 7, 8};
    DenseMatrix             values(3, 3);
    values(0, 0) = 1.0;
    values(1, 1) = 1.0;
    values(2, 2) = 1.0;

    out.begin(pattern_);
    out.addElement(
        {rows.view(), rows.view(), entries.view(), values.view()});
    out.finalize();
  }

private:
  HostCsrPattern pattern_;
};

bool near(Real a, Real b)
{
  return std::abs(a - b) <= 1.0e-12;
}

template <std::size_t N>
bool valsNear(const HostVector<Real>&    actual,
              const std::array<Real, N>& expected)
{
  if (actual.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[static_cast<Index>(i)], expected[i]))
    {
      return false;
    }
  }
  return true;
}

fem::DirichletControl mappedControl()
{
  return fem::DirichletControl(
      HostVector<Index>{0, 2},
      1,
      HostVector<fem::DirichletControlMapEntry>{{0, 0, 2.0},
                                                {1, 0, -1.0}});
}

TestOutcome mappedTimeDirichletResidual()
{
  TestStatus status(__func__);

  DenseMatrix modes(3, 1);
  modes(1, 0)        = 3.0;
  const auto initial = fem::makeInitialStateMap(
      HostVector<Real>{1.0, 2.0, 3.0},
      std::move(modes),
      mappedControl(),
      0,
      0,
      2);
  const IdentityTimeResidual                  base;
  const assembly::HostConstrainedTimeResidual res(
      base,
      fem::makeControlMap(
          2,
          3,
          mappedControl(),
          {},
          {},
          HostVector<LinearInterpolation>{{0, 1, 0.25}, {1, 1, 0.0}},
          0),
      initial);

  status *= res.dims().num_param == 2;

  const HostVector<Real>       history{1.0, 2.0, 3.0};
  const HostVector<Real>       next{10.0, 20.0, 30.0};
  const HostVector<Real>       parameters{4.0, 8.0};
  const state::HostTimeContext ctx{
      0,
      next.view(),
      parameters.view(),
      state::HostTimeHistoryView(history.data(), 1, 3)};
  linalg::HostContext cpu;

  HostVector<Real> out;
  res.initialState(parameters.view(), out, cpu);
  status *= valsNear(out, std::array<Real, 3>{{8.0, 14.0, -4.0}});
  const HostVector<Real> initial_adj{1.0, 2.0, 3.0};
  HostVector<Real>       initial_grad(2);
  res.addInitialStateJacT(
      initial_adj.view(), initial_grad.view(), cpu);
  status *= valsNear(initial_grad, std::array<Real, 2>{{5.0, 0.0}});

  linalg::HostJacobian jac(cpu);
  res.assembleNext(ctx, out, jac, cpu);
  status *= valsNear(out, std::array<Real, 3>{{0.0, 20.0, 35.0}});

  const HostVector<Real> adj{1.0, 5.0, 3.0};
  res.applyJacT(ctx, state::VariableBlock::Param, adj.view(), out, cpu);
  status *= valsNear(out, std::array<Real, 2>{{0.75, 0.25}});

  res.applyJacT(
      ctx, state::VariableBlock::hist(0), adj.view(), out, cpu);
  status *= valsNear(out, std::array<Real, 3>{{0.0, 0.0, 0.0}});

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::mappedTimeDirichletResidual();

  return results.summary();
}
