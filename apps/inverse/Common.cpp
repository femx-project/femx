#include "Common.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace std;

namespace femx::inverse
{

string requireValue(int           argc,
                    char**        argv,
                    int&          i,
                    const string& key)
{
  if (i + 1 >= argc)
  {
    throw runtime_error("Missing value for " + key);
  }
  return string(argv[++i]);
}

void ensureParentDir(const string& path)
{
  const filesystem::path out(path);
  const filesystem::path dir = out.parent_path();
  if (!dir.empty())
  {
    filesystem::create_directories(dir);
  }
}

void writeVector(const string&       path,
                 const Vector<Real>& x,
                 VectorFileHeader    header)
{
  ensureParentDir(path);
  ofstream out(path);
  if (!out)
  {
    throw runtime_error("Failed to open output file: " + path);
  }

  out << setprecision(17);
  if (header == VectorFileHeader::size)
  {
    out << x.size() << '\n';
  }
  for (Real value : x)
  {
    out << value << '\n';
  }
}

void writeMatrix(const string&      path,
                 const DenseMatrix& mat)
{
  ensureParentDir(path);
  ofstream out(path);
  if (!out)
  {
    throw runtime_error("Failed to open output file: " + path);
  }

  out << setprecision(17);
  out << mat.rows() << ' ' << mat.cols() << '\n';
  for (Index i = 0; i < mat.rows(); ++i)
  {
    for (Index j = 0; j < mat.cols(); ++j)
    {
      if (j > 0)
      {
        out << ' ';
      }
      out << mat(i, j);
    }
    out << '\n';
  }
}

} // namespace femx::inverse
