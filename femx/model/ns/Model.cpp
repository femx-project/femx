#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/Assembly.hpp>
#include <femx/assembly/ConstrainedTimeResidual.hpp>
#include <femx/common/Checks.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/fem/elements/LagrangeTetrahedronP1.hpp>
#include <femx/fem/elements/LagrangeTriangleP1.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaJacobian.hpp>
#include <femx/linalg/native/HostContext.hpp>

namespace femx::model::ns
{
namespace
{

constexpr Index kQuadratureOrder = 2;

std::unique_ptr<fem::FiniteElement> makeElement(const fem::Mesh& mesh)
{
  require(mesh.numElems() > 0, "Mesh has no elements");

  const fem::Element::Shape shape = mesh.elems().front().shape();
  if (shape == fem::Element::Shape::Quadrilateral)
  {
    return std::make_unique<fem::LagrangeQuadQ1>();
  }
  if (shape == fem::Element::Shape::Triangle)
  {
    return std::make_unique<fem::LagrangeTriangleP1>();
  }
  if (shape == fem::Element::Shape::Tetrahedron)
  {
    return std::make_unique<fem::LagrangeTetrahedronP1>();
  }
  throw std::runtime_error("Unsupported Navier-Stokes mesh element type");
}

fem::MixedFESpace makeSpace(fem::Mesh& mesh, fem::FiniteElement& elem)
{
  fem::FESpace velocity_space(&mesh, &elem, mesh.dim());
  fem::FESpace pressure_space(&mesh, &elem);

  fem::MixedFESpace space;
  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();
  return space;
}

void requireModelPrm(Index nstep, Real dt, const FluidProperties& fluid)
{
  require(nstep > 0,
          "NavierStokesModel requires a positive number of time steps");
  require(dt > 0.0 && std::isfinite(dt),
          "NavierStokesModel requires a positive finite time step");
  require(std::isfinite(fluid.rho) && fluid.rho > 0.0,
          "NavierStokesModel requires positive finite density");
  require(std::isfinite(fluid.mu) && fluid.mu > 0.0,
          "NavierStokesModel requires positive finite viscosity");
}

fem::Mesh validatedModelMesh(fem::Mesh              mesh,
                             Index                  nstep,
                             Real                   dt,
                             const FluidProperties& fluid)
{
  requireModelPrm(nstep, dt, fluid);
  return mesh;
}

fem::Mesh readModelMesh(const std::string&     path,
                        Index                  nstep,
                        Real                   dt,
                        const FluidProperties& fluid)
{
  requireModelPrm(nstep, dt, fluid);
  require(!path.empty(), "NavierStokesModel mesh file is required");
  return fem::GmshReader::read(path);
}

fem::GaussQuadrature makeVelocityQuadrature(
    const fem::MixedFESpace& space)
{
  return fem::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(),
      kQuadratureOrder);
}

fem::HostElementQuadratureData makeNavierElementData(
    const fem::MixedFESpace& space)
{
  auto data = fem::makeElementQuadratureData(
      space.field(0).space(), makeVelocityQuadrature(space));
  const Index num_dofs = (data.dim() + 1) * data.numShapes();
  require(data.numElems() > 0 && data.numQuadraturePoints() > 0
              && data.numShapes() > 0 && data.dim() > 0
              && data.dim() <= kMaxDim
              && data.numQuadraturePoints() <= kMaxNq
              && data.numShapes() <= kMaxNn && num_dofs <= kMaxNd,
          "Navier element quadrature data has unsupported dimensions");
  return data;
}

void add(HostVector<Real>& vec, Index i, Real val)
{
#pragma omp atomic update
  vec[i] += val;
}

} // namespace

struct NavierWork
{
  HostVector<Real> hist;
  HostVector<Real> nxt;
  HostVector<Real> adj;
  HostVector<Real> vjp;
};

void gather(const assembly::HostAssemblyMap& map,
            Index                            num_hist,
            HostVectorView<const Real>       hist,
            HostVectorView<const Real>       nxt,
            Index                            ie,
            NavierWork&                      work)
{
  const auto  map_v = map.view();
  const Index nc    = map_v.numStateDofs(ie);
  work.hist.resize(num_hist * nc);
  work.nxt.resize(nc);
  for (Index lag = 0; lag < num_hist; ++lag)
  {
    for (Index col = 0; col < nc; ++col)
    {
      work.hist[lag * nc + col] =
          hist[lag * map.numStates() + map_v.stateDof(ie, col)];
    }
  }
  for (Index col = 0; col < nc; ++col)
  {
    work.nxt[col] = nxt[map_v.stateDof(ie, col)];
  }
}

assembly::HostTimeElementView elem(Index             step,
                                   Index             num_hist,
                                   Index             ie,
                                   const NavierWork& work)
{
  return {ie, step, num_hist, work.hist.view(), work.nxt.view()};
}

void reduce(HostVector<Real>& vec,
            Index,
            Index,
            Index,
            linalg::Context<MemorySpace::Host>& ctx)
{
  ctx.allReduceSum(vec.view());
}

namespace detail
{

template <class Ctx>
void applyHistJacT(
    const HostElementKernel&         kernel,
    Index                            step,
    Index                            num_hist,
    Index                            lag,
    Index                            ie_begin,
    Index                            ie_end,
    const assembly::HostAssemblyMap& map,
    HostVectorView<const Real>       hist,
    HostVectorView<const Real>       nxt,
    HostVectorView<const Real>       adj,
    HostVector<Real>&                out,
    Ctx&                             ctx)
{
  ctx.vectors().resizeOrZero(out, map.numStates());
#pragma omp parallel
  {
    NavierWork work;
#pragma omp for
    for (Index ie = ie_begin; ie < ie_end; ++ie)
    {
      gather(map, num_hist, hist, nxt, ie, work);
      const auto  e     = elem(step, num_hist, ie, work);
      const auto  map_v = map.view();
      const Index nr    = map_v.numResDofs(ie);
      const Index nc    = map_v.numStateDofs(ie);
      work.adj.resize(nr);
      work.vjp.resize(num_hist * nc);
      for (Index row = 0; row < nr; ++row)
      {
        work.adj[row] = adj[map_v.resDof(ie, row)];
      }
      histVjp(kernel, e, work.adj.view(), work.vjp.view());
      for (Index col = 0; col < nc; ++col)
      {
        add(out,
            map_v.stateDof(ie, col),
            work.vjp[lag * nc + col]);
      }
    }
  }
  reduce(out, ie_begin, ie_end, map.numElems(), ctx);
}

} // namespace detail

template <MemorySpace Space>
class NavierResidual : public state::TimeResidual<Space>
{
public:
  using Base      = state::TimeResidual<Space>;
  using Vec       = typename Base::Vec;
  using ConstView = typename Base::ConstView;
  using Jac       = typename Base::Jac;
  using Ctx       = typename Base::Ctx;
  using StepCtx   = typename Base::StepCtx;
  using Map       = assembly::AssemblyMap<Space>;
  using Data      = fem::ElementQuadratureData<Space>;
  using Kernel    = ElementKernel<Space>;

  NavierResidual(Index                            nstep,
                 const assembly::HostAssemblyMap& map,
                 HostElementKernel                kernel)
    : nstep_(nstep)
  {
    if constexpr (Space == MemorySpace::Host)
    {
      map_ptr_      = &map;
      host_pattern_ = &map.pattern();
      kernel_       = kernel;
    }
    else
    {
      require(false, "Host Navier residual requires Host storage");
    }
  }

  NavierResidual(Index                                 nstep,
                 const assembly::HostAssemblyMap&      map,
                 const fem::HostElementQuadratureData& data,
                 FluidProperties                       fluid,
                 Real                                  dt,
                 Ctx&                                  ctx)
    : nstep_(nstep)
  {
    if constexpr (Space == MemorySpace::Device)
    {
      auto& cuda_ctx = dynamic_cast<linalg::CudaContext&>(ctx);
      owned_map_     = std::make_unique<Map>();
      copy(map, *owned_map_, cuda_ctx);
      owned_data_ = std::make_unique<Data>();
      fem::copy(data, *owned_data_, cuda_ctx);
      host_pattern_store_ = map.pattern();
      map_ptr_            = owned_map_.get();
      host_pattern_       = &host_pattern_store_;
      kernel_             = Kernel(owned_data_->view(), fluid, dt);
    }
    else
    {
      require(false, "Device Navier residual requires Device storage");
    }
  }

  state::TimeDims dims() const override
  {
    return {nstep_, map().numStates(), 0, map().numRes(), kNumHist};
  }

  const HostCsrPattern& hostPattern() const override
  {
    return *host_pattern_;
  }

  void initialState(ConstView prm, Vec& out, Ctx& ctx) const override
  {
    require(prm.empty(), "Navier physics residual is parameter-free");
    auto& vec_handler = ctx.vectors();
    vec_handler.resizeOrZero(out, map().numStates());
  }

  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Jac&           jac,
                    Ctx&           ctx) const override
  {
    checkCtx(time);
    const auto      range = ctx.elementRange(map().numElems());
    const ConstView hist{time.hist.data(), kNumHist * map().numStates()};
    if constexpr (Space == MemorySpace::Device)
    {
      auto& cuda_ctx = dynamic_cast<linalg::CudaContext&>(ctx);
      auto& cuda_jac = dynamic_cast<linalg::CudaJacobian&>(jac);
      detail::assembleNext(kernel_,
                           time.step,
                           kNumHist,
                           range.begin,
                           range.end,
                           map(),
                           hist,
                           time.nxt,
                           res,
                           cuda_jac,
                           cuda_ctx);
    }
    else
    {
      assembly::assemble(kernel_,
                         time.step,
                         kNumHist,
                         state::VariableBlock::NextState,
                         map(),
                         range.begin,
                         range.end,
                         hist,
                         time.nxt,
                         res,
                         jac,
                         ctx);
    }
  }

  void applyJacT(const StepCtx&       time,
                 state::VariableBlock wrt,
                 ConstView            adj,
                 Vec&                 out,
                 Ctx&                 ctx) const override
  {
    checkCtx(time);
    require(!wrt.isNextState(),
            "Navier transpose apply supports only history and parameter blocks");
    require(adj.size() == map().numRes(),
            "Navier residual adjoint size mismatch");
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    require(wrt.historyLag() >= 0 && wrt.historyLag() < kNumHist,
            "Navier residual history lag is out of range");
    if (!ad::has_enzyme)
    {
      throw std::runtime_error(
          "Navier history VJP requires Enzyme. Configure with "
          "-DFEMX_ENABLE_ENZYME=ON and provide Enzyme_DIR.");
    }

    const ConstView hist{time.hist.data(), kNumHist * map().numStates()};
    const auto      range = ctx.elementRange(map().numElems());
    if constexpr (Space == MemorySpace::Device)
    {
      auto& cuda_ctx = dynamic_cast<linalg::CudaContext&>(ctx);
      detail::applyHistJacT(kernel_,
                            time.step,
                            kNumHist,
                            wrt.historyLag(),
                            range.begin,
                            range.end,
                            map(),
                            hist,
                            time.nxt,
                            adj,
                            out,
                            cuda_ctx);
    }
    else
    {
      detail::applyHistJacT(kernel_,
                            time.step,
                            kNumHist,
                            wrt.historyLag(),
                            range.begin,
                            range.end,
                            map(),
                            hist,
                            time.nxt,
                            adj,
                            out,
                            ctx);
    }
  }

private:
  const Map& map() const
  {
    return *map_ptr_;
  }

  void checkCtx(const StepCtx& ctx) const
  {
    require(ctx.step >= 0 && ctx.step < nstep_,
            "Navier residual step is out of range");
    require(ctx.hist.count() >= kNumHist
                && ctx.hist.stateSize() == map().numStates()
                && ctx.nxt.size() == map().numStates() && ctx.prm.empty(),
            "Navier residual vector size mismatch");
  }

  Index                 nstep_{0};
  std::unique_ptr<Map>  owned_map_;
  std::unique_ptr<Data> owned_data_;
  const Map*            map_ptr_{nullptr};
  HostCsrPattern        host_pattern_store_;
  const HostCsrPattern* host_pattern_{nullptr};
  Kernel                kernel_;
};

class NavierStokesModel::Residual final
  : public NavierResidual<MemorySpace::Host>
{
public:
  using NavierResidual::NavierResidual;
};

NavierStokesModel::NavierStokesModel(const std::string& path,
                                     Index              nstep,
                                     Real               dt,
                                     FluidProperties    fluid)
  : NavierStokesModel(
        readModelMesh(path, nstep, dt, fluid),
        nstep,
        dt,
        fluid)
{
}

NavierStokesModel::NavierStokesModel(fem::Mesh       mesh,
                                     Index           nstep,
                                     Real            dt,
                                     FluidProperties fluid)
  : nstep_(nstep),
    dt_(dt),
    mesh_(validatedModelMesh(std::move(mesh), nstep_, dt_, fluid)),
    element_(makeElement(mesh_)),
    space_(makeSpace(mesh_, *element_)),
    geometry_(fem::makeGeometry(mesh_)),
    fluid_(fluid),
    data_(makeNavierElementData(space_)),
    map_(assembly::makeAssemblyMap(fem::DofLayout(space_)))
{
  res_ = std::make_unique<Residual>(nstep_, map_, elementKernel());
}

NavierStokesModel::~NavierStokesModel() = default;

Index NavierStokesModel::numSteps() const
{
  return nstep_;
}

Index NavierStokesModel::numStates() const
{
  return space_.numDofs();
}

Real NavierStokesModel::dt() const
{
  return dt_;
}

const FluidProperties& NavierStokesModel::fluid() const
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

state::HostTimeResidual& NavierStokesModel::residual()
{
  return *res_;
}

const state::HostTimeResidual& NavierStokesModel::residual() const
{
  return *res_;
}

const assembly::HostAssemblyMap& NavierStokesModel::map() const
{
  return map_;
}

const fem::HostElementQuadratureData& NavierStokesModel::data() const
{
  return data_;
}

HostElementKernel NavierStokesModel::elementKernel() const
{
  return {data_.view(), {fluid_.rho, fluid_.mu}, dt_};
}

#if defined(FEMX_HAS_CUDA)
std::unique_ptr<state::DeviceTimeResidual> makeDeviceTimeResidual(
    const NavierStokesModel& model,
    fem::HostControlMap      control,
    fem::HostInitialStateMap init_state)
{
  linalg::CudaContext ctx;
  auto                base = std::make_unique<NavierResidual<MemorySpace::Device>>(
      model.numSteps(),
      model.map(),
      model.data(),
      FluidProperties{model.fluid().rho, model.fluid().mu},
      model.dt(),
      ctx);
  auto out = std::make_unique<assembly::DeviceConstrainedTimeResidual>(
      std::move(base), std::move(control), std::move(init_state), ctx);
  ctx.sync();
  return out;
}
#endif

HostVector<Index> NavierStokesModel::velocityDofs() const
{
  const auto  velocity  = space_.field(0);
  const Index num_nodes = mesh_.numNodes();
  const Index num_comps = velocity.numComponents();

  HostVector<Index> dofs;
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

HostVector<Index> NavierStokesModel::velocityBoundaryDofs(
    Index boundary_tag) const
{
  return fem::makeVelocityControl(space_, boundary_tag).stateDofs();
}

HostVector<Index> NavierStokesModel::velocityBoundaryDofs(
    const std::string& boundary_name) const
{
  return fem::makeVelocityControl(space_, boundary_name).stateDofs();
}

} // namespace femx::model::ns
