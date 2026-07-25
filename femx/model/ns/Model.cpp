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
#include <femx/linalg/handler/VectorHandler.hpp>

#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>
#endif

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

void add(HostVector& vec, Index i, Real val)
{
#pragma omp atomic update
  vec[i] += val;
}

void resizeOrZero(HostVector& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    std::fill(out.begin(), out.end(), Real{});
  }
}

#if defined(FEMX_HAS_PETSC)
void allreduce(HostVector& vec, MPI_Comm comm)
{
  const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                 vec.data(),
                                 static_cast<int>(vec.size()),
                                 MPIU_REAL,
                                 MPI_SUM,
                                 comm);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error("Navier residual MPI_Allreduce failed");
  }
}
#endif

} // namespace

struct NavierWork
{
  HostVector hist;
  HostVector nxt;
  HostVector adj;
  HostVector vjp;
};

void gather(const assembly::HostAssemblyMap& map,
            Index                            num_hist,
            HostConstVectorView              hist,
            HostConstVectorView              nxt,
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

void reduce(HostVector&,
            Index,
            Index,
            Index,
            CpuContext&)
{
}

#if defined(FEMX_HAS_PETSC)
void reduce(HostVector& vec,
            Index,
            Index,
            Index,
            linalg::PetscContext& ctx)
{
  int       comm_size = 0;
  const int ierr      = MPI_Comm_size(ctx.comm, &comm_size);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error(
        "Navier history VJP communicator query failed");
  }
  if (comm_size > 1)
  {
    allreduce(vec, ctx.comm);
  }
}
#endif

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
    HostConstVectorView              hist,
    HostConstVectorView              nxt,
    HostConstVectorView              adj,
    HostVector&                      out,
    Ctx&                             ctx)
{
  resizeOrZero(out, map.numStates());
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

template <class Backend>
class NavierResidual : public state::TimeResidual<Backend>
{
public:
  using Base      = state::TimeResidual<Backend>;
  using Vec       = typename Base::Vec;
  using ConstView = typename Base::ConstView;
  using Mat       = typename Base::Mat;
  using Pattern   = typename Base::Pattern;
  using Ctx       = typename Base::Ctx;
  using StepCtx   = typename Base::StepCtx;
  using Map       = assembly::AssemblyMap<Backend::space>;
  using Data      = fem::ElementQuadratureData<Backend::space>;
  using Kernel    = ElementKernel<Backend::space>;

  NavierResidual(Index                            nstep,
                 const assembly::HostAssemblyMap& map,
                 HostElementKernel                kernel)
    : nstep_(nstep)
  {
    if constexpr (Backend::space == MemorySpace::Host)
    {
      map_ptr_      = &map;
      host_pattern_ = &map.pattern();
      kernel_       = kernel;
      ie_end_       = map.numElems();
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
    if constexpr (Backend::space == MemorySpace::Device)
    {
      owned_map_ = std::make_unique<Map>();
      copy(map, *owned_map_, ctx);
      owned_data_ = std::make_unique<Data>();
      fem::copy(data, *owned_data_, ctx);
      host_pattern_store_ = map.pattern();
      map_ptr_            = owned_map_.get();
      host_pattern_       = &host_pattern_store_;
      kernel_             = Kernel(owned_data_->view(), fluid, dt);
      ie_end_             = map.numElems();
    }
    else
    {
      require(false, "Device Navier residual requires Device storage");
    }
  }

  void setElemRange(Index ie_begin, Index ie_end)
  {
    require(ie_begin >= 0 && ie_end >= ie_begin
                && ie_end <= map().numElems(),
            "Navier residual element range is invalid");
    if constexpr (Backend::space == MemorySpace::Device)
    {
      require(ie_begin == 0 && ie_end == map().numElems(),
              "CUDA Navier residual requires the full element range");
    }
#if !defined(FEMX_HAS_PETSC)
    else
    {
      require(ie_begin == 0 && ie_end == map().numElems(),
              "Navier residual element ranges require PETSc");
    }
#endif
    ie_begin_ = ie_begin;
    ie_end_   = ie_end;
  }

  state::TimeDims dims() const override
  {
    return {nstep_, map().numStates(), 0, map().numRes(), kNumHist};
  }

  const HostCsrPattern& hostPattern() const override
  {
    return *host_pattern_;
  }

  const Pattern& pattern() const override
  {
    return map().pattern();
  }

  void initialState(ConstView prm, Vec& out, Ctx& ctx) const override
  {
    require(prm.empty(), "Navier physics residual is parameter-free");
    linalg::VectorHandler<Backend> vec_handler(ctx);
    vec_handler.resizeOrZero(out, map().numStates());
  }

  void assembleNext(const StepCtx& time,
                    Vec&           res,
                    Mat&           jac,
                    Ctx&           ctx) const override
  {
    checkCtx(time);
    const ConstView hist{time.hist.data(), kNumHist * map().numStates()};
    if constexpr (Backend::space == MemorySpace::Device)
    {
      detail::assembleNext(kernel_,
                           time.step,
                           kNumHist,
                           ie_begin_,
                           ie_end_,
                           map(),
                           hist,
                           time.nxt,
                           res,
                           jac,
                           ctx);
    }
    else
    {
      assembly::assemble(kernel_,
                         time.step,
                         kNumHist,
                         state::VariableBlock::NextState,
                         map(),
                         ie_begin_,
                         ie_end_,
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
    detail::applyHistJacT(kernel_,
                          time.step,
                          kNumHist,
                          wrt.historyLag(),
                          ie_begin_,
                          ie_end_,
                          map(),
                          hist,
                          time.nxt,
                          adj,
                          out,
                          ctx);
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
  Index                 ie_begin_{0};
  Index                 ie_end_{0};
};

class NavierStokesModel::Residual final
  : public NavierResidual<linalg::HostCsrBackend>
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
  ie_end_ = map_.numElems();
  res_    = std::make_unique<Residual>(nstep_, map_, elementKernel());
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

void NavierStokesModel::setElemRange(Index ie_begin, Index ie_end)
{
  ie_begin_ = ie_begin;
  ie_end_   = ie_end;
  res_->setElemRange(ie_begin, ie_end);
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
  CudaContext ctx;
  auto        base = std::make_unique<NavierResidual<linalg::CudaCsrBackend>>(
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

#if defined(FEMX_HAS_PETSC)
std::unique_ptr<state::TimeResidual<linalg::PetscBackend>>
makePetscTimeResidual(const NavierStokesModel& model)
{
  auto res = std::make_unique<NavierResidual<linalg::PetscBackend>>(
      model.numSteps(), model.map(), model.elementKernel());
  res->setElemRange(model.ie_begin_, model.ie_end_);
  return res;
}
#endif

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
