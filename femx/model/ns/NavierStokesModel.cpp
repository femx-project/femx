#include "NavierStokesModel.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/assembly/Assembly.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/model/ns/Helper.hpp>

#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>
#endif

namespace femx::model::ns
{
namespace
{

constexpr Index kQuadratureOrder = 2;

void requireModelParameters(Index              num_steps,
                            Real               dt,
                            const FluidParams& fluid)
{
  if (num_steps <= 0)
  {
    throw std::runtime_error(
        "NavierStokesModel requires a positive number of time steps");
  }
  if (dt <= 0.0 || !std::isfinite(dt))
  {
    throw std::runtime_error(
        "NavierStokesModel requires a positive finite time step");
  }
  if (!std::isfinite(fluid.rho) || fluid.rho <= 0.0)
  {
    throw std::runtime_error(
        "NavierStokesModel requires positive finite density");
  }
  if (!std::isfinite(fluid.mu) || fluid.mu <= 0.0)
  {
    throw std::runtime_error(
        "NavierStokesModel requires positive finite viscosity");
  }
}

fem::Mesh validatedModelMesh(fem::Mesh          mesh,
                             Index              num_steps,
                             Real               dt,
                             const FluidParams& fluid)
{
  requireModelParameters(num_steps, dt, fluid);
  return mesh;
}

fem::Mesh readModelMesh(const std::string& mesh_file,
                        Index              num_steps,
                        Real               dt,
                        const FluidParams& fluid)
{
  requireModelParameters(num_steps, dt, fluid);
  if (mesh_file.empty())
  {
    throw std::runtime_error("NavierStokesModel mesh file is required");
  }
  return fem::GmshReader::read(mesh_file);
}

fem::GaussQuadrature makeVelocityQuadrature(
    const fem::MixedFESpace& space)
{
  return fem::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadratureOrder);
}

void add(HostVector& vec, Index i, Real val)
{
#pragma omp atomic update
  vec[i] += val;
}

#if defined(FEMX_HAS_PETSC)
void allreduce(HostVector& vec)
{
  const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                 vec.data(),
                                 static_cast<int>(vec.size()),
                                 MPIU_REAL,
                                 MPI_SUM,
                                 PETSC_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error("Navier residual MPI_Allreduce failed");
  }
}
#endif

} // namespace

class NavierStokesModel::Residual final : public state::TimeResidual
{
public:
  Residual(Index                             nstep,
           const assembly::HostAssemblyMap&  map,
           NavierOperator<MemorySpace::Host> op)
    : nstep_(nstep), map_(map), op_(op), ie_end_(map.numElems())
  {
  }

  void setElemRange(Index ie_begin, Index ie_end)
  {
    if (ie_begin < 0 || ie_end < ie_begin || ie_end > map_.numElems())
    {
      throw std::runtime_error("Navier residual element range is invalid");
    }
#if !defined(FEMX_HAS_PETSC)
    if (ie_begin != 0 || ie_end != map_.numElems())
    {
      throw std::runtime_error(
          "Navier residual element ranges require PETSc");
    }
#endif
    ie_begin_ = ie_begin;
    ie_end_   = ie_end;
  }

  state::TimeDims dims() const override
  {
    return {nstep_, map_.numStates(), 0, map_.numRes(), kNumHist};
  }

  void res(const state::TimeContext& ctx,
           HostVector&               out) const override
  {
    checkCtx(ctx);
    resizeOrZero(out, map_.numRes());

#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index nc  = map.numStateDofs(ie);
        work.jac.resize(nc);
        for (Index row = 0; row < nr; ++row)
        {
          Real val = 0.0;
          op_.evalRow(e,
                      state::VariableBlock::NextState,
                      row,
                      val,
                      work.jac.view());
          add(out, map.resDof(ie, row), val);
        }
      }
    }
    reduce(out);
  }

  void applyJac(const state::TimeContext& ctx,
                state::VariableBlock      wrt,
                const HostVector&         dir,
                HostVector&               out) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    if (dir.size() != (wrt.isParam() ? 0 : map_.numStates()))
    {
      throw std::runtime_error("Navier residual direction size mismatch");
    }

    resizeOrZero(out, map_.numRes());
    if (wrt.isParam())
    {
      return;
    }

#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index nc  = map.numStateDofs(ie);
        work.jac.resize(nc);
        for (Index row = 0; row < nr; ++row)
        {
          Real unused = 0.0;
          op_.evalRow(e, wrt, row, unused, work.jac.view());
          Real val = 0.0;
          for (Index col = 0; col < nc; ++col)
          {
            val += work.jac[col] * dir[map.stateDof(ie, col)];
          }
          add(out, map.resDof(ie, row), val);
        }
      }
    }
    reduce(out);
  }

  void applyJacT(const state::TimeContext& ctx,
                 state::VariableBlock      wrt,
                 const HostVector&         adj,
                 HostVector&               out) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    if (adj.size() != map_.numRes())
    {
      throw std::runtime_error("Navier residual adjoint size mismatch");
    }
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }

    resizeOrZero(out, map_.numStates());
#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index nc  = map.numStateDofs(ie);
        work.jac.resize(nc);
        for (Index row = 0; row < nr; ++row)
        {
          Real unused = 0.0;
          op_.evalRow(e, wrt, row, unused, work.jac.view());
          const Real val = adj[map.resDof(ie, row)];
          for (Index col = 0; col < nc; ++col)
          {
            add(out,
                map.stateDof(ie, col),
                work.jac[col] * val);
          }
        }
      }
    }
    reduce(out);
  }

  bool assembleJac(const state::TimeContext& ctx,
                   state::VariableBlock      wrt,
                   linalg::MatrixOperator&   out) const override
  {
    checkCtx(ctx);
    checkWrt(wrt);
    const Index nc = wrt.isParam() ? 0 : map_.numStates();
    out.resize(map_.numRes(), nc);
    out.setZero();
    if (wrt.isParam())
    {
      return true;
    }

#pragma omp parallel
    {
      Work work;
#pragma omp for
      for (Index ie = ie_begin_; ie < ie_end_; ++ie)
      {
        gather(ctx, ie, work);
        const auto  e   = elem(ctx.step, ie, work);
        const auto  map = map_.view();
        const Index nr  = map.numResDofs(ie);
        const Index ne  = map.numStateDofs(ie);
        work.mat.resize(nr, ne);
        work.rows.resize(nr);
        work.cols.resize(ne);

        for (Index row = 0; row < nr; ++row)
        {
          work.rows[row] = map.resDof(ie, row);
          Real unused    = 0.0;
          op_.evalRow(e,
                      wrt,
                      row,
                      unused,
                      {work.mat.data() + row * ne, ne});
        }
        for (Index col = 0; col < ne; ++col)
        {
          work.cols[col] = map.stateDof(ie, col);
        }
        out.addElem(ie, work.rows, work.cols, work.mat, true);
      }
    }
    return true;
  }

private:
  static constexpr Index kNumHist = 2;

  struct Work
  {
    HostVector   hist;
    HostVector   nxt;
    HostVector   jac;
    DenseMatrix  mat;
    Array<Index> rows;
    Array<Index> cols;
  };

  void checkCtx(const state::TimeContext& ctx) const
  {
    if (ctx.step < 0 || ctx.step >= nstep_)
    {
      throw std::runtime_error("Navier residual step is out of range");
    }
    if (ctx.hist.count() < kNumHist
        || ctx.hist.stateSize() != map_.numStates()
        || ctx.nxt == nullptr || ctx.nxt->size() != map_.numStates()
        || ctx.prm == nullptr || !ctx.prm->empty())
    {
      throw std::runtime_error("Navier residual vector size mismatch");
    }
  }

  static void checkWrt(state::VariableBlock wrt)
  {
    if (wrt.isHistoryState()
        && (wrt.historyLag() < 0 || wrt.historyLag() >= kNumHist))
    {
      throw std::runtime_error(
          "Navier residual history lag is out of range");
    }
  }

  void gather(const state::TimeContext& ctx,
              Index                     ie,
              Work&                     work) const
  {
    const auto  map = map_.view();
    const Index nc  = map.numStateDofs(ie);
    work.hist.resize(kNumHist * nc);
    work.nxt.resize(nc);
    for (Index lag = 0; lag < kNumHist; ++lag)
    {
      const auto src = ctx.hist.state(lag);
      for (Index col = 0; col < nc; ++col)
      {
        work.hist[lag * nc + col] = src[map.stateDof(ie, col)];
      }
    }
    for (Index col = 0; col < nc; ++col)
    {
      work.nxt[col] = (*ctx.nxt)[map.stateDof(ie, col)];
    }
  }

  static assembly::TimeElementView<MemorySpace::Host> elem(
      Index       step,
      Index       ie,
      const Work& work)
  {
    return {ie, step, kNumHist, work.hist.view(), work.nxt.view()};
  }

  void reduce(HostVector& vec) const
  {
#if defined(FEMX_HAS_PETSC)
    if (ie_begin_ != 0 || ie_end_ != map_.numElems())
    {
      allreduce(vec);
    }
#else
    (void) vec;
#endif
  }

  Index                             nstep_{0};
  const assembly::HostAssemblyMap&  map_;
  NavierOperator<MemorySpace::Host> op_;
  Index                             ie_begin_{0};
  Index                             ie_end_{0};
};

NavierStokesModel::NavierStokesModel(const std::string& mesh_file,
                                     Index              num_steps,
                                     Real               dt,
                                     FluidParams        fluid)
  : NavierStokesModel(
        readModelMesh(mesh_file, num_steps, dt, fluid),
        num_steps,
        dt,
        fluid)
{
}

NavierStokesModel::NavierStokesModel(fem::Mesh   mesh,
                                     Index       num_steps,
                                     Real        dt,
                                     FluidParams fluid)
  : num_steps_(num_steps),
    dt_(dt),
    mesh_(validatedModelMesh(std::move(mesh), num_steps_, dt_, fluid)),
    element_(makeElement(mesh_)),
    space_(makeSpace(mesh_, *element_)),
    geometry_(fem::makeGeometry(mesh_)),
    fluid_(fluid),
    data_(makeNavierData(space_.field(0).space(),
                         makeVelocityQuadrature(space_))),
    map_(assembly::makeAssemblyMap(fem::DofLayout(space_)))
{
  res_ = std::make_unique<Residual>(num_steps_, map_, op());
}

NavierStokesModel::~NavierStokesModel() = default;

Index NavierStokesModel::numSteps() const
{
  return num_steps_;
}

Index NavierStokesModel::numStates() const
{
  return space_.numDofs();
}

Real NavierStokesModel::dt() const
{
  return dt_;
}

const FluidParams& NavierStokesModel::fluid() const
{
  return fluid_;
}

const fem::Mesh& NavierStokesModel::mesh() const
{
  return mesh_;
}

const fem::MixedFESpace& NavierStokesModel::space() const
{
  return space_;
}

const fem::HostGeometry& NavierStokesModel::geometry() const
{
  return geometry_;
}

state::TimeResidual& NavierStokesModel::residual()
{
  return *res_;
}

const state::TimeResidual& NavierStokesModel::residual() const
{
  return *res_;
}

void NavierStokesModel::setElemRange(Index ie_begin, Index ie_end)
{
  res_->setElemRange(ie_begin, ie_end);
}

const assembly::HostAssemblyMap& NavierStokesModel::map() const
{
  return map_;
}

const HostNavierData& NavierStokesModel::data() const
{
  return data_;
}

NavierOperator<MemorySpace::Host> NavierStokesModel::op() const
{
  return {data_.view(), {fluid_.rho, fluid_.mu}, dt_};
}

Array<Index> NavierStokesModel::velocityDofs() const
{
  const auto  velocity  = space_.field(0);
  const Index num_nodes = mesh_.numNodes();
  const Index num_comps = velocity.numComponents();

  Array<Index> dofs;
  dofs.reserve(num_nodes * num_comps);
  for (Index node = 0; node < num_nodes; ++node)
  {
    for (Index component = 0; component < num_comps; ++component)
    {
      dofs.push_back(velocity.globalDof(node, component));
    }
  }
  return dofs;
}

Array<Index> NavierStokesModel::velocityBoundaryDofs(
    Index boundary_tag) const
{
  return fem::makeVelocityControl(space_, boundary_tag).stateDofs();
}

Array<Index> NavierStokesModel::velocityBoundaryDofs(
    const std::string& boundary_name) const
{
  return fem::makeVelocityControl(space_, boundary_name).stateDofs();
}

} // namespace femx::model::ns
