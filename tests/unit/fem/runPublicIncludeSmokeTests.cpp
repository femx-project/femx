#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/FEMResidual.hpp>
#include <femx/assembly/Kernel.hpp>
#include <femx/fem/BoundaryCondition.hpp>
#include <femx/fem/Cell.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/Quadrature.hpp>

int main()
{
  femx::Mesh              mesh;
  femx::Cell*             cell       = nullptr;
  femx::Quadrature*       quadrature = nullptr;
  femx::DofLayout*        layout     = nullptr;
  femx::BoundaryCondition boundary;

  femx::assembly::Kernel*       kernel   = nullptr;
  femx::assembly::Assembler*    assembler = nullptr;
  femx::assembly::FEMResidual*  residual = nullptr;

  (void)cell;
  (void)quadrature;
  (void)layout;
  (void)boundary;
  (void)kernel;
  (void)assembler;
  (void)residual;

  return mesh.numNodes() == 0 ? 0 : 1;
}
