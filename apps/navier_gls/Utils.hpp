#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Navier.hpp"
#include <refem/io/TimeSeriesDataOut.hpp>
#include <refem/linalg/Vector.hpp>
#include <refem/mesh/Mesh.hpp>
#include <refem/solver/ReSolverLinearSolver.hpp>

struct Options
{
  std::string backend = "cpu";
};

Options parseOptions(int argc, char* argv[])
{
  Options options;

  const auto requireValue = [argc, argv](int& i, const std::string& key)
  {
    if (i + 1 >= argc)
    {
      throw std::runtime_error("Missing value for " + key);
    }
    return std::string(argv[++i]);
  };

  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      std::cout << "Usage: navierstokes [--nx N] [--ny N] [--steps N] "
                   "[--dt DT] [--mu MU] [--rho RHO] [--lid U] "
                   "[--interval N] [--max-cfl N] "
                   "[-b|--backend cpu|cuda]\n";
      std::exit(0);
    }
    else if (key == "-b" || key == "--backend")
    {
      options.backend = requireValue(i, key);
    }
    else if (key == "--nx")
    {
      refem::nx =
          static_cast<refem::index_type>(std::stol(requireValue(i, key)));
    }
    else if (key == "--ny")
    {
      refem::ny =
          static_cast<refem::index_type>(std::stol(requireValue(i, key)));
    }
    else if (key == "--steps")
    {
      refem::steps =
          static_cast<refem::index_type>(std::stol(requireValue(i, key)));
    }
    else if (key == "--interval")
    {
      refem::interval =
          static_cast<refem::index_type>(std::stol(requireValue(i, key)));
    }
    else if (key == "--dt")
    {
      refem::dt = std::stod(requireValue(i, key));
    }
    else if (key == "--mu")
    {
      refem::mu = std::stod(requireValue(i, key));
    }
    else if (key == "--rho")
    {
      refem::rho = std::stod(requireValue(i, key));
    }
    else if (key == "--lid")
    {
      refem::lid = std::stod(requireValue(i, key));
    }
    else if (key == "--max-cfl")
    {
      refem::max_cfl = std::stod(requireValue(i, key));
    }
    else
    {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (refem::nx <= 0 || refem::ny <= 0 || refem::steps <= 0 ||
      refem::dt <= 0.0 || refem::max_cfl <= 0.0 || refem::rho <= 0.0 ||
      refem::mu <= 0.0)
  {
    throw std::runtime_error("Invalid non-positive Navier-Stokes parameter");
  }

  refem::interval = std::max<refem::index_type>(1, refem::interval);

  if (options.backend != "cpu" && options.backend != "cuda")
  {
    throw std::runtime_error("Backend must be either 'cpu' or 'cuda'");
  }

  return options;
}

bool isFinite(const refem::Vector& x)
{
  for (refem::index_type i = 0; i < x.size(); ++i)
  {
    if (!std::isfinite(x[i]))
    {
      return false;
    }
  }
  return true;
}

void setSolverOptions(refem::ReSolveOptions& options)
{
  options.factor             = "none";
  options.refactor           = "none";
  options.ir                 = "none";
  options.max_iterations     = 5000;
  options.restart            = 200;
  options.relative_tolerance = 1.0e-10;
  options.solve              = "fgmres";
  options.precond            = "ilu0";
  options.flexible           = true;
}

void writeTimeSeriesOutput(const refem::Mesh&                  mesh,
                           const std::vector<refem::Snapshot>& snapshots)
{
  const std::string root = std::string(REFEM_NAVIERSTOKES_OUTPUT_DIR);

  refem::TimeSeriesDataOut velocity_out;
  velocity_out.attachMesh(mesh);

  refem::TimeSeriesDataOut pressure_out;
  pressure_out.attachMesh(mesh);

  for (const refem::Snapshot& snapshot : snapshots)
  {
    velocity_out.beginStep(snapshot.time);
    velocity_out.addNodalVectorField("velocity", snapshot.ux, snapshot.uy);

    pressure_out.beginStep(snapshot.time);
    pressure_out.addNodalScalarField("pressure", snapshot.p);
  }

  velocity_out.write(root + "/velocity");
  pressure_out.write(root + "/pressure");
}
