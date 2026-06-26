#include <stdexcept>

#include <femx/fem/Mesh.hpp>

using namespace std;

namespace femx
{

Mesh Mesh::makeStructuredQuad(Index nx,
                              Index ny,
                              Real  x_min,
                              Real  x_max,
                              Real  y_min,
                              Real  y_max)
{
  Mesh mesh(2);

  const Real dx = (x_max - x_min) / static_cast<Real>(nx);
  const Real dy = (y_max - y_min) / static_cast<Real>(ny);

  for (Index j = 0; j <= ny; ++j)
  {
    const Real y = y_min + static_cast<Real>(j) * dy;
    for (Index i = 0; i <= nx; ++i)
    {
      const Real x = x_min + static_cast<Real>(i) * dx;
      mesh.addNode({x, y, 0.0});
    }
  }

  const Index nodes_per_row = nx + 1;
  for (Index j = 0; j < ny; ++j)
  {
    for (Index i = 0; i < nx; ++i)
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
