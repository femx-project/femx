#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "TestHelper.hpp"
#include <femx/fem/Mesh.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/io/VtiWriter.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace tests
{
namespace
{

std::string readFile(const std::string& fname)
{
  std::ifstream in(fname, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("Failed to open test output: " + fname);
  }
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

bool contains(const std::string& haystack, const std::string& needle)
{
  return haystack.find(needle) != std::string::npos;
}

TestOutcome vtuWriterWritesMeshPointData()
{
  TestStatus status(__func__);

  const std::string fname = "femx_io_unit_mesh.vtu";
  std::remove(fname.c_str());

  const Mesh   mesh = Mesh::makeStructuredQuad(1, 1);
  Vector<Real> values{0.0, 1.0, 2.0, 3.0};

  VtuWriter writer;
  writer.writePointData(fname,
                        mesh,
                        Vector<VtuWriter::PointField>{
                            {"u&v", 1, &values}});

  const std::string text  = readFile(fname);
  status                 *= contains(text, "<VTKFile type=\"UnstructuredGrid\"");
  status                 *= contains(text, "NumberOfPoints=\"4\"");
  status                 *= contains(text, "NumberOfCells=\"1\"");
  status                 *= contains(text, "Name=\"u&amp;v\"");
  status                 *= contains(text, "Name=\"connectivity\"");
  status                 *= std::remove(fname.c_str()) == 0;

  bool threw = false;
  try
  {
    writer.writePointData("femx_io_unit_bad.vtu",
                          mesh,
                          Vector<VtuWriter::PointField>{
                              {"bad", 1, nullptr}});
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;
  std::remove("femx_io_unit_bad.vtu");

  return status.report();
}

TestOutcome vtiWriterWritesImageCellData()
{
  TestStatus status(__func__);

  const std::string fname = "femx_io_unit_image.vti";
  std::remove(fname.c_str());

  VtiWriter::Image image;
  image.elem_counts = {2, 1, 1};
  image.origin      = {1.0, 2.0, 3.0};
  image.spacing     = {0.5, 0.25, 1.0};
  image.time        = 1.25;

  Vector<Real> vals{4.0, 5.0};

  VtiWriter writer;
  writer.writeElemData(fname,
                       image,
                       Vector<VtiWriter::ElemField>{
                           {"cell<value>", 1, &vals}});

  const std::string text  = readFile(fname);
  status                 *= contains(text, "<VTKFile type=\"ImageData\"");
  status                 *= contains(text, "WholeExtent=\"0 2 0 0 0 0\"");
  status                 *= contains(text, "Origin=\"1 2 3\"");
  status                 *= contains(text, "Spacing=\"0.5 0.25 1\"");
  status                 *= contains(text, "Name=\"TimeValue\"");
  status                 *= contains(text, "Name=\"cell&lt;value&gt;\"");
  status                 *= std::remove(fname.c_str()) == 0;

  bool threw = false;
  try
  {
    writer.writeElemData("femx_io_unit_bad.vti",
                         image,
                         Vector<VtiWriter::ElemField>{
                             {"bad", 1, nullptr}});
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;
  std::remove("femx_io_unit_bad.vti");

  return status.report();
}

TestOutcome timeSeriesDataOutValidatesInputs()
{
  TestStatus status(__func__);

  const Mesh mesh = Mesh::makeStructuredQuad(1, 1);

  TimeSeriesDataOut out;
  bool              threw = false;
  try
  {
    out.addNodalScalarField("u", Vector<Real>{1.0});
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  out.attachMesh(mesh);
  out.beginStep(0.0);
  out.addNodalScalarField("u", Vector<Real>{1.0, 2.0});

  threw = false;
  try
  {
    out.write("femx_io_unit_timeseries");
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  std::remove("femx_io_unit_timeseries.xdmf");
  std::remove("femx_io_unit_timeseries.h5");

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::vtuWriterWritesMeshPointData();
  results += femx::tests::vtiWriterWritesImageCellData();
  results += femx::tests::timeSeriesDataOutValidatesInputs();

  return results.summary();
}
