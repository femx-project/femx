#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Assembly.hpp"
#include "NavierStokesEquation.hpp"
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{

using Clock = std::chrono::steady_clock;

struct Options
{
  femx::Index nx             = 64;
  femx::Index ny             = 16;
  femx::Index local_repeats  = 20;
  femx::Index global_repeats = 5;
};

Options parseOptions(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg       = argv[i];
    auto              readIndex = [&](femx::Index& value)
    {
      if (i + 1 >= argc)
      {
        throw std::runtime_error("Missing value for " + arg);
      }
      value = static_cast<femx::Index>(std::atoll(argv[++i]));
    };

    if (arg == "-nx")
    {
      readIndex(options.nx);
    }
    else if (arg == "-ny")
    {
      readIndex(options.ny);
    }
    else if (arg == "-local_repeats")
    {
      readIndex(options.local_repeats);
    }
    else if (arg == "-global_repeats")
    {
      readIndex(options.global_repeats);
    }
    else
    {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  return options;
}

femx::MixedFESpace makeSpace(femx::Mesh& mesh, femx::LagrangeQuadQ1& elem)
{
  femx::FESpace velocity_space(&mesh, &elem, mesh.dim());
  femx::FESpace pressure_space(&mesh, &elem);

  femx::MixedFESpace space;
  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();
  return space;
}

void fillState(const femx::MixedFESpace& space, femx::Vector<femx::Real>& x)
{
  x.resize(space.numDofs());
  x.setZero();

  const auto  u_dof = space.field(0);
  const auto  p_dof = space.field(1);
  const auto& mesh  = space.mesh();

  for (femx::Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto&      node     = mesh.node(in);
    const femx::Real y        = node[1];
    const femx::Real profile  = 1.5 * (1.0 - (2.0 * y - 1.0) * (2.0 * y - 1.0));
    x[u_dof.globalDof(in, 0)] = profile;
    x[u_dof.globalDof(in, 1)] = 0.02 * std::sin(2.0 * node[0]);
    x[p_dof.globalDof(in, 0)] = 1.0 - 0.2 * node[0];
  }
}

double secondsSince(Clock::time_point start, Clock::time_point end)
{
  return std::chrono::duration<double>(end - start).count();
}

} // namespace

int main(int argc, char** argv)
{
  const Options options = parseOptions(argc, argv);

#if !defined(FEMX_HAS_ENZYME)
  std::cerr << "runTimeNavierStokesAssemblyBenchmark requires "
               "FEMX_ENABLE_ENZYME=ON\n";
  return 77;
#endif

  femx::Mesh mesh =
      femx::Mesh::makeStructuredQuad(options.nx, options.ny, 0.0, 4.0, 0.0, 1.0);
  femx::LagrangeQuadQ1 elem;
  femx::MixedFESpace   space = makeSpace(mesh, elem);

  femx::TimeNavierStokesParameters prm;
  prm.steps      = 40;
  prm.dt         = 0.025;
  prm.fluid.rho  = 1.0;
  prm.fluid.mu   = 0.015;
  prm.quad_order = 2;

  femx::Vector<femx::Real> x;
  femx::Vector<femx::Real> x_next;
  femx::Vector<femx::Real> prm;
  fillState(space, x);
  fillState(space, x_next);

  const auto quad = femx::GaussQuadrature::make(
      space.field(0).space().finiteElement().referenceElement(), prm.quad_order);

  volatile femx::Real local_checksum = 0.0;
  const auto          local_start    = Clock::now();
  for (femx::Index repeat = 0; repeat < options.local_repeats; ++repeat)
  {
    femx::Real repeat_checksum = 0.0;
#pragma omp parallel reduction(+ : repeat_checksum)
    {
      femx::ElementValues        ev(elem, quad);
      femx::DenseMatrix          Ke(space.numDofsPerElem(), space.numDofsPerElem());

#pragma omp for
      for (femx::Index ic = 0; ic < space.numElems(); ++ic)
      {
        femx::assemblePrevElemMatrix(
            space, ic, ev, quad, x_next, x, prm, Ke);
        repeat_checksum += Ke(0, 0) + Ke(Ke.rows() - 1, Ke.cols() - 1);
      }
    }
    local_checksum += repeat_checksum;
  }
  const auto local_end = Clock::now();

  femx::NavierStokesEquation eq(space, prm);
  const auto                             pattern = femx::assembly::SparsityPatternBuilder::build(space);
  femx::system::SparseSystemMatrix       prev_jac(pattern);

  volatile femx::Real global_checksum = 0.0;
  const auto          global_start    = Clock::now();
  for (femx::Index repeat = 0; repeat < options.global_repeats; ++repeat)
  {
    eq.assemblePrevStateJac(0, x_next, x, prm, prev_jac);
    global_checksum += prev_jac.matrix().valuesData()[0];
  }
  const auto global_end = Clock::now();

  const double local_seconds  = secondsSince(local_start, local_end);
  const double global_seconds = secondsSince(global_start, global_end);
  const double local_cell_visits =
      static_cast<double>(options.local_repeats) * static_cast<double>(space.numElems());

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "backend: enzyme\n";
  std::cout << "mesh: " << options.nx << " x " << options.ny
            << ", cells: " << space.numElems()
            << ", states: " << space.numDofs()
            << ", elem dofs: " << space.numDofsPerElem() << '\n';
#if defined(_OPENMP)
  std::cout << "threads: " << omp_get_max_threads() << '\n';
#else
  std::cout << "threads: 1\n";
#endif
  std::cout << "local repeats: " << options.local_repeats
            << ", seconds: " << local_seconds
            << ", us/cell: " << 1.0e6 * local_seconds / local_cell_visits
            << ", checksum: " << local_checksum << '\n';
  std::cout << "global repeats: " << options.global_repeats
            << ", seconds: " << global_seconds
            << ", ms/assembly: "
            << 1.0e3 * global_seconds / static_cast<double>(options.global_repeats)
            << ", checksum: " << global_checksum << '\n';

  return 0;
}
