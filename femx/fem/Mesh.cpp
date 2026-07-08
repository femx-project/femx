#include <stdexcept>

#include <femx/fem/Mesh.hpp>

using namespace std;

namespace femx
{

Mesh Mesh::makeStructuredQuad(Index num_x_cells,
                              Index num_y_cells,
                              Real  x_min,
                              Real  x_max,
                              Real  y_min,
                              Real  y_max)
{
  Mesh mesh(2);

  const Real dx = (x_max - x_min) / static_cast<Real>(num_x_cells);
  const Real dy = (y_max - y_min) / static_cast<Real>(num_y_cells);

  for (Index j = 0; j <= num_y_cells; ++j)
  {
    const Real y = y_min + static_cast<Real>(j) * dy;
    for (Index i = 0; i <= num_x_cells; ++i)
    {
      const Real x = x_min + static_cast<Real>(i) * dx;
      mesh.addNode({x, y, 0.0});
    }
  }

  const Index nodes_per_row = num_x_cells + 1;
  for (Index j = 0; j < num_y_cells; ++j)
  {
    for (Index i = 0; i < num_x_cells; ++i)
    {
      const Index n0 = j * nodes_per_row + i;
      const Index n1 = n0 + 1;
      const Index n3 = n0 + nodes_per_row;
      const Index n2 = n3 + 1;
      mesh.addElem({n0, n1, n2, n3},
                   Element::Shape::Quadrilateral,
                   2,
                   0,
                   0,
                   {});
    }
  }

  return mesh;
}

} // namespace femx
