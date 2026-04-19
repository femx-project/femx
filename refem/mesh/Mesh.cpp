#include <stdexcept>

#include <refem/mesh/Mesh.hpp>

namespace refem
{

Mesh Mesh::makeStructuredQuad(index_type nx,
                              index_type ny,
                              real_type  x_min,
                              real_type  x_max,
                              real_type  y_min,
                              real_type  y_max)
{
  Mesh mesh(2);

  const real_type dx = (x_max - x_min) / static_cast<real_type>(nx);
  const real_type dy = (y_max - y_min) / static_cast<real_type>(ny);

  for (index_type j = 0; j <= ny; ++j)
  {
    const real_type y = y_min + static_cast<real_type>(j) * dy;
    for (index_type i = 0; i <= nx; ++i)
    {
      const real_type x = x_min + static_cast<real_type>(i) * dx;
      mesh.addNode({x, y, 0.0});
    }
  }

  const index_type nodes_per_row = nx + 1;
  for (index_type j = 0; j < ny; ++j)
  {
    for (index_type i = 0; i < nx; ++i)
    {
      const index_type n0 = j * nodes_per_row + i;
      const index_type n1 = n0 + 1;
      const index_type n3 = n0 + nodes_per_row;
      const index_type n2 = n3 + 1;
      mesh.addCell({n0, n1, n2, n3});
    }
  }

  return mesh;
}

} // namespace refem
